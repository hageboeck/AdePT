// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#include "TestEm3.cuh"

#include <AdePT/BVHNavigator.h>

#include <CopCore/PhysicalConstants.h>

#include <G4HepEmGammaManager.hh>
#include <G4HepEmGammaTrack.hh>
#include <G4HepEmTrack.hh>
#include <G4HepEmGammaInteractionCompton.hh>
#include <G4HepEmGammaInteractionConversion.hh>
#include <G4HepEmGammaInteractionPhotoelectric.hh>
// Pull in implementation.
#include <G4HepEmGammaManager.icc>
#include <G4HepEmGammaInteractionCompton.icc>
#include <G4HepEmGammaInteractionConversion.icc>
#include <G4HepEmGammaInteractionPhotoelectric.icc>

__global__ __launch_bounds__(ThreadsPerBlock, MinBlocksPerSM)
void TransportGammas(Track *gammas, const adept::MParray *active, Secondaries secondaries,
                     adept::MParray *activeQueue, GlobalScoring *globalScoring,
                     ScoringPerVolume *scoringPerVolume,
                     SOAData * soaData)
{
  int activeSize = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot       = (*active)[i];
    Track &currentTrack  = gammas[slot];
    const int volumeID   = currentTrack.navState.Top()->id();
    const int theMCIndex = MCIndex[volumeID];

    auto survive = [&] {
      activeQueue->push_back(slot);
    };
    
    // Ensure that this track doesn't undergo an interaction unless we overwrite this:
    assert(i < SOAData::queueSize);
    soaData->nextInteraction[i] = -1;

    // Init a track with the needed data to call into G4HepEm.
    G4HepEmGammaTrack gammaTrack;
    G4HepEmTrack *theTrack = gammaTrack.GetTrack();
    theTrack->SetEKin(currentTrack.energy);
    theTrack->SetMCIndex(theMCIndex);

    // Sample the `number-of-interaction-left` and put it into the track.
    for (int ip = 0; ip < 3; ++ip) {
      double numIALeft = currentTrack.numIALeft[ip];
      if (numIALeft <= 0) {
        numIALeft                  = -std::log(currentTrack.Uniform());
        currentTrack.numIALeft[ip] = numIALeft;
      }
      theTrack->SetNumIALeft(numIALeft, ip);
    }

    // Call G4HepEm to compute the physics step limit.
    G4HepEmGammaManager::HowFar(&g4HepEmData, &g4HepEmPars, &gammaTrack);

    // Get result into variables.
    double geometricalStepLengthFromPhysics = theTrack->GetGStepLength();
    int winnerProcessIndex                  = theTrack->GetWinnerProcessIndex();
    // Leave the range and MFP inside the G4HepEmTrack. If we split kernels, we
    // also need to carry them over!

    // Check if there's a volume boundary in between.
    vecgeom::NavStateIndex nextState;
    double geometryStepLength = BVHNavigator::ComputeStepAndNextVolume(
        currentTrack.pos, currentTrack.dir, geometricalStepLengthFromPhysics, currentTrack.navState, nextState);
    currentTrack.pos += geometryStepLength * currentTrack.dir;
    atomicAdd(&globalScoring->neutralSteps, 1);

    // Set boundary state in navState so the next step and secondaries get the
    // correct information (currentTrack.navState = nextState only if relocated
    // in case of a boundary; see below)
    currentTrack.navState.SetBoundaryState(nextState.IsOnBoundary());

    // Propagate information from geometrical step to G4HepEm.
    theTrack->SetGStepLength(geometryStepLength);
    theTrack->SetOnBoundary(nextState.IsOnBoundary());

    G4HepEmGammaManager::UpdateNumIALeft(theTrack);

    // Save the `number-of-interaction-left` in our track.
    for (int ip = 0; ip < 3; ++ip) {
      double numIALeft           = theTrack->GetNumIALeft(ip);
      currentTrack.numIALeft[ip] = numIALeft;
    }

    if (nextState.IsOnBoundary()) {
      // For now, just count that we hit something.
      atomicAdd(&globalScoring->hits, 1);

      // Kill the particle if it left the world.
      if (nextState.Top() != nullptr) {
        BVHNavigator::RelocateToNextVolume(currentTrack.pos, currentTrack.dir, nextState);

        // Move to the next boundary.
        currentTrack.navState = nextState;
        survive();
      }
      continue;
    } else if (winnerProcessIndex < 0) {
      // No discrete process, move on.
      survive();
      continue;
    }

    // Reset number of interaction left for the winner discrete process.
    // (Will be resampled in the next iteration.)
    currentTrack.numIALeft[winnerProcessIndex] = -1.0;

    soaData->nextInteraction[i] = winnerProcessIndex;
  }
}


template<int ProcessIndex>
__global__ __launch_bounds__(ThreadsPerBlock, MinBlocksPerSM)
void ComputeGammaInteractions(Track *gammas, const adept::MParray *active, Secondaries secondaries,
                              adept::MParray *activeQueue, GlobalScoring *globalScoring,
                              ScoringPerVolume *scoringPerVolume, SOAData const * soaData)
{
  constexpr int sharedSize = 12250;
  __shared__ int candidates[sharedSize];
  __shared__ unsigned int counter;
  __shared__ unsigned int noopCounter;
  counter = 0;
  noopCounter = 0;

  const int activeSize = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const auto winnerProcess = soaData->nextInteraction[i];

    if (winnerProcess == ProcessIndex) {
      const auto destination = atomicInc(&counter, (unsigned int)-1);
      candidates[destination % sharedSize] = i;
    } else atomicInc(&noopCounter, (unsigned int)-1);
  }

  __syncthreads();
  assert(counter < sharedSize);
  if (threadIdx.x == 0 && counter >= sharedSize) {
    printf("Error: Shared queue for %d exhausted to %d in %s:%d\n", ProcessIndex, counter, __FILE__, __LINE__);
    __threadfence();
    asm("trap;");
  }


  for (int i = threadIdx.x; i < counter; i += blockDim.x) {
    const int slot = (*active)[candidates[i]];
    Track &currentTrack = gammas[slot];
    const auto volume         = currentTrack.navState.Top();
    const int volumeID        = volume->id();
    const int theMCIndex      = MCIndex[volumeID];
    const double energy       = currentTrack.energy;

    RanluxppDouble newRNG{currentTrack.rngState.Branch()};
    RanluxppDoubleEngine rnge{&currentTrack.rngState};


    if constexpr (ProcessIndex == 0) {
      // Invoke gamma conversion to e-/e+ pairs, if the energy is above the threshold.
      if (energy < 2 * copcore::units::kElectronMassC2) {
        survive();
        continue;
      }

      double logEnergy = std::log(energy);
      double elKinEnergy, posKinEnergy;
      G4HepEmGammaInteractionConversion::SampleKinEnergies(&g4HepEmData, energy, logEnergy, theMCIndex, elKinEnergy,
                                                          posKinEnergy, &rnge);

      double dirPrimary[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
      double dirSecondaryEl[3], dirSecondaryPos[3];
      G4HepEmGammaInteractionConversion::SampleDirections(dirPrimary, dirSecondaryEl, dirSecondaryPos, elKinEnergy,
                                                          posKinEnergy, &rnge);

      Track &electron = secondaries.electrons.NextTrack();
      Track &positron = secondaries.positrons.NextTrack();
      atomicAdd(&globalScoring->numElectrons, 1);
      atomicAdd(&globalScoring->numPositrons, 1);

      electron.InitAsSecondary(/*parent=*/currentTrack);
      electron.rngState = newRNG;
      electron.energy = elKinEnergy;
      electron.dir.Set(dirSecondaryEl[0], dirSecondaryEl[1], dirSecondaryEl[2]);

      positron.InitAsSecondary(/*parent=*/currentTrack);
      // Reuse the RNG state of the dying track.
      positron.rngState = currentTrack.rngState;
      positron.energy = posKinEnergy;
      positron.dir.Set(dirSecondaryPos[0], dirSecondaryPos[1], dirSecondaryPos[2]);

      // The current track is killed by not enqueuing into the next activeQueue.
    } else if constexpr (ProcessIndex == 1) {
      // Invoke Compton scattering of gamma.
      constexpr double LowEnergyThreshold = 100 * copcore::units::eV;
      if (energy < LowEnergyThreshold) {
        survive();
        continue;
      }
      const double origDirPrimary[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
      double dirPrimary[3];
      const double newEnergyGamma =
          G4HepEmGammaInteractionCompton::SamplePhotonEnergyAndDirection(energy, dirPrimary, origDirPrimary, &rnge);
      vecgeom::Vector3D<double> newDirGamma(dirPrimary[0], dirPrimary[1], dirPrimary[2]);

      const double energyEl = energy - newEnergyGamma;
      if (energyEl > LowEnergyThreshold) {
        // Create a secondary electron and sample/compute directions.
        Track &electron = secondaries.electrons.NextTrack();
        atomicAdd(&globalScoring->numElectrons, 1);

        electron.InitAsSecondary(/*parent=*/currentTrack);
        electron.rngState = newRNG;
        electron.energy = energyEl;
        electron.dir    = energy * currentTrack.dir - newEnergyGamma * newDirGamma;
        electron.dir.Normalize();
      } else {
        atomicAdd(&globalScoring->energyDeposit, energyEl);
        atomicAdd(&scoringPerVolume->energyDeposit[volumeID], energyEl);
      }

      // Check the new gamma energy and deposit if below threshold.
      if (newEnergyGamma > LowEnergyThreshold) {
        currentTrack.energy = newEnergyGamma;
        currentTrack.dir    = newDirGamma;
        survive();
      } else {
        atomicAdd(&globalScoring->energyDeposit, newEnergyGamma);
        atomicAdd(&scoringPerVolume->energyDeposit[volumeID], newEnergyGamma);
        // The current track is killed by not enqueuing into the next activeQueue.
      }
    } else if constexpr (ProcessIndex == 2) {
      // Invoke photoelectric process.
      const double theLowEnergyThreshold = 1 * copcore::units::eV;

      const double bindingEnergy = G4HepEmGammaInteractionPhotoelectric::SelectElementBindingEnergy(
          &g4HepEmData, theMCIndex, currentTrack.fPEmxSec, energy, &rnge);

      double edep             = bindingEnergy;
      const double photoElecE = energy - edep;
      if (photoElecE > theLowEnergyThreshold) {
        // Create a secondary electron and sample directions.
        Track &electron = secondaries.electrons.NextTrack();
        atomicAdd(&globalScoring->numElectrons, 1);

        double dirGamma[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
        double dirPhotoElec[3];
        G4HepEmGammaInteractionPhotoelectric::SamplePhotoElectronDirection(photoElecE, dirGamma, dirPhotoElec, &rnge);

        electron.InitAsSecondary(/*parent=*/currentTrack);
        electron.rngState = newRNG;
        electron.energy   = photoElecE;
        electron.dir.Set(dirPhotoElec[0], dirPhotoElec[1], dirPhotoElec[2]);
      } else {
        edep = energy;
      }
      atomicAdd(&globalScoring->energyDeposit, edep);
      atomicAdd(&scoringPerVolume->energyDeposit[volumeID], edep);
      // The current track is killed by not enqueuing into the next activeQueue.
    }
  }
}


template
__global__ __launch_bounds__(ThreadsPerBlock, MinBlocksPerSM)
void ComputeGammaInteractions<0>(Track *gammas, const adept::MParray *active, Secondaries secondaries,
                                adept::MParray *activeQueue, GlobalScoring *globalScoring,
                                ScoringPerVolume *scoringPerVolume,
                                SOAData const * soaData);
template
__global__ __launch_bounds__(ThreadsPerBlock, MinBlocksPerSM)
void ComputeGammaInteractions<1>(Track *gammas, const adept::MParray *active, Secondaries secondaries,
                                adept::MParray *activeQueue, GlobalScoring *globalScoring,
                                ScoringPerVolume *scoringPerVolume,
                                SOAData const * soaData);
template
__global__ __launch_bounds__(ThreadsPerBlock, MinBlocksPerSM)
void ComputeGammaInteractions<2>(Track *gammas, const adept::MParray *active, Secondaries secondaries,
                                adept::MParray *activeQueue, GlobalScoring *globalScoring,
                                ScoringPerVolume *scoringPerVolume,
                                SOAData const * soaData);
