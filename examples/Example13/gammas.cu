// SPDX-FileCopyrightText: 2021 CERN
// SPDX-License-Identifier: Apache-2.0

#include "example13.cuh"

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

__device__
void ComputePhysicsStepLimit(Track &track)
{
  int id = track.navState.Top()->GetLogicalVolume()->id();

  G4HepEmTrack* t = track.gammaTrack.GetTrack();

  t->SetEKin(track.energy);
  t->SetMCIndex(MCIndex[id]);

  for (int ip = 0; ip < 3; ++ip)
    if (t->GetNumIALeft(ip) <= 0)
      t->SetNumIALeft(-std::log(track.Uniform()), ip);

  G4HepEmGammaManager::HowFar(&g4HepEmData, &g4HepEmPars, &track.gammaTrack);
}

__global__ void ComputePhysicsStepLimit(Track *gammas, const adept::MParray *active)
{
  int activeSize = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x)
    ComputePhysicsStepLimit(gammas[(*active)[i]]);
}

__device__
bool ComputeGeometryStepAndPropagate(Track &track)
{
#ifdef VECGEOM_FLOAT_PRECISION
  const Precision kPush = 10 * vecgeom::kTolerance;
#else
  const Precision kPush = 0.;
#endif
  vecgeom::NavStateIndex nextState;
  G4HepEmTrack* theTrack = track.gammaTrack.GetTrack();

  double StepLength = BVHNavigator::ComputeStepAndNextVolume(track.pos, track.dir,
    theTrack->GetGStepLength(), track.navState, nextState, kPush);

  track.pos += StepLength * track.dir;

  // Propagate information from geometrical step to G4HepEm.
  theTrack->SetGStepLength(StepLength);
  theTrack->SetOnBoundary(nextState.IsOnBoundary());
  G4HepEmGammaManager::UpdateNumIALeft(theTrack);

  // Relocate track if necessary, else set boundary state to propagate
  // information correctly to secondaries and the next step
  track.navState.SetBoundaryState(nextState.IsOnBoundary());

  if (nextState.IsOnBoundary() && nextState.Top()) {
    BVHNavigator::RelocateToNextVolume(track.pos, track.dir, nextState);
    track.navState = nextState;
  }

  return nextState.Top() != nullptr; /* returns if particle is still inside world */
}

__global__ void ComputeGeometryStepAndPropagate(Track *gammas, const adept::MParray *active,
                                                adept::MParray *activeQueue, GlobalScoring *globalScoring)
{
  int activeSize = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    int slot      = (*active)[i];
    Track &currentTrack = gammas[slot];

    bool inWorld = ComputeGeometryStepAndPropagate(currentTrack);
    bool onBoundary = currentTrack.navState.IsOnBoundary();

    currentTrack.done = onBoundary && inWorld;

    if (onBoundary) {
      // For now, just count that we hit something.
      atomicAdd(&globalScoring->hits, 1);

      // Kill the particle if it left the world.
      if (inWorld)
        activeQueue->push_back(slot);
    }

    atomicAdd(&globalScoring->neutralSteps, 1);
  }
}

__device__ bool GammaConversion(Track &track, Secondaries &secondaries,
                                GlobalScoring *globalScoring, ScoringPerVolume *)
{
  // Return if energy is below threshold
  if (track.energy < 2 * copcore::units::kElectronMassC2)
    return false;

  double logEnergy = std::log(track.energy);
  double elKinEnergy, posKinEnergy;

  RanluxppDoubleEngine rnge(&track.rngState);
  G4HepEmTrack* theTrack = track.gammaTrack.GetTrack();

  G4HepEmGammaInteractionConversion::SampleKinEnergies(&g4HepEmData, track.energy, logEnergy, theTrack->GetMCIndex(), elKinEnergy,
      posKinEnergy, &rnge);

  double dirPrimary[] = {track.dir.x(), track.dir.y(), track.dir.z()};
  double dirSecondaryEl[3], dirSecondaryPos[3];
  G4HepEmGammaInteractionConversion::SampleDirections(dirPrimary, dirSecondaryEl, dirSecondaryPos, elKinEnergy,
      posKinEnergy, &rnge);

  Track &electron = secondaries.electrons.NextTrack();
  Track &positron = secondaries.positrons.NextTrack();

  electron.InitAsSecondary(/*parent=*/track);
  electron.rngState = track.rngState.Branch();
  electron.energy   = elKinEnergy;
  electron.dir.Set(dirSecondaryEl[0], dirSecondaryEl[1], dirSecondaryEl[2]);

  positron.InitAsSecondary(/*parent=*/track);
  // Reuse the RNG state of the dying track.
  positron.rngState = track.rngState;
  positron.energy   = posKinEnergy;
  positron.dir.Set(dirSecondaryPos[0], dirSecondaryPos[1], dirSecondaryPos[2]);

  atomicAdd(&globalScoring->numElectrons, 1);
  atomicAdd(&globalScoring->numPositrons, 1);

  return true;
}

__device__ bool ComptonScattering(Track &track, Secondaries &secondaries,
                                  GlobalScoring *globalScoring, ScoringPerVolume *scoringPerVolume)
{
  constexpr double LowEnergyThreshold = 100 * copcore::units::eV;

  if (track.energy < LowEnergyThreshold)
    return false;

  RanluxppDoubleEngine rnge(&track.rngState);
  int volumeID = track.navState.Top()->id();

  const double origDirPrimary[] = {track.dir.x(), track.dir.y(), track.dir.z()};
  double dirPrimary[3];
  const double newEnergyGamma =
    G4HepEmGammaInteractionCompton::SamplePhotonEnergyAndDirection(track.energy, dirPrimary, origDirPrimary, &rnge);
  vecgeom::Vector3D<double> newDirGamma(dirPrimary[0], dirPrimary[1], dirPrimary[2]);

  const double energyEl = track.energy - newEnergyGamma;

  if (energyEl > LowEnergyThreshold) {
    // Create a secondary electron and sample/compute directions.
    Track &electron = secondaries.electrons.NextTrack();
    atomicAdd(&globalScoring->numElectrons, 1);

    electron.InitAsSecondary(/*parent=*/track);
    electron.rngState = track.rngState.Branch();
    electron.energy   = energyEl;
    electron.dir      = track.energy * track.dir - newEnergyGamma * newDirGamma;
    electron.dir.Normalize();
  } else {
    atomicAdd(&globalScoring->energyDeposit, energyEl);
    atomicAdd(&scoringPerVolume->energyDeposit[volumeID], energyEl);
  }

  // Check the new gamma energy and deposit if below threshold.
  if (newEnergyGamma > LowEnergyThreshold) {
    track.energy = newEnergyGamma;
    track.dir    = newDirGamma;

    // The current track continues to live.
    return false;
  } else {
    atomicAdd(&globalScoring->energyDeposit, newEnergyGamma);
    atomicAdd(&scoringPerVolume->energyDeposit[volumeID], newEnergyGamma);
    return true;
  }
}

__device__ bool PhotoElectricEffect(Track &track, Secondaries &secondaries,
                                    GlobalScoring *globalScoring, ScoringPerVolume *scoringPerVolume)
{
  // Invoke photoelectric process.
  RanluxppDoubleEngine rnge(&track.rngState);
  const double theLowEnergyThreshold = 1 * copcore::units::eV;

  const double bindingEnergy = G4HepEmGammaInteractionPhotoelectric::SelectElementBindingEnergy(
      &g4HepEmData, track.gammaTrack.GetTrack()->GetMCIndex(), track.gammaTrack.GetPEmxSec(), track.energy, &rnge);

  double edep             = bindingEnergy;
  const double photoElecE = track.energy - edep;
  if (photoElecE > theLowEnergyThreshold) {
    // Create a secondary electron and sample directions.
    Track &electron = secondaries.electrons.NextTrack();
    atomicAdd(&globalScoring->numElectrons, 1);

    double dirGamma[] = {track.dir.x(), track.dir.y(), track.dir.z()};
    double dirPhotoElec[3];
    G4HepEmGammaInteractionPhotoelectric::SamplePhotoElectronDirection(photoElecE, dirGamma, dirPhotoElec, &rnge);

    electron.InitAsSecondary(/*parent=*/track);
    electron.rngState = track.rngState.Branch();
    electron.energy   = photoElecE;
    electron.dir.Set(dirPhotoElec[0], dirPhotoElec[1], dirPhotoElec[2]);
  } else {
    edep = track.energy;
  }
  atomicAdd(&globalScoring->energyDeposit, edep);
  return true;
}

__global__ void ComputeInteraction(Track *gammas, const adept::MParray *active, Secondaries secondaries,
                                adept::MParray *activeQueue, GlobalScoring *globalScoring,
                                ScoringPerVolume *scoringPerVolume)
{
  int activeSize = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot = (*active)[i];
    Track &currentTrack = gammas[slot];

    // Skip tracks that have hit a boundary
    if (currentTrack.done)
      continue;

    G4HepEmTrack* theTrack = currentTrack.gammaTrack.GetTrack();
    int winnerProcessIndex = theTrack->GetWinnerProcessIndex();

    // Skip tracks that have no active discrete process
    if (winnerProcessIndex < 0) {
      activeQueue->push_back(slot);
      continue;
    }

    // Reset number of interaction left. It will be resampled in the next iteration.
    theTrack->SetNumIALeft(-1, theTrack->GetWinnerProcessIndex());

    // Perform the discrete interaction.
    switch (winnerProcessIndex) {
    case 0: {
      if (!GammaConversion(currentTrack, secondaries, globalScoring, scoringPerVolume))
        activeQueue->push_back(slot);
      break;
    }
    case 1: {
      if (!ComptonScattering(currentTrack, secondaries, globalScoring, scoringPerVolume))
        activeQueue->push_back(slot);
      break;
    }
    case 2: {
      PhotoElectricEffect(currentTrack, secondaries, globalScoring, scoringPerVolume);
      break;
    }
    }
  }
}

__global__ void TransportGammas(Track *gammas, const adept::MParray *active, Secondaries secondaries,
                                adept::MParray *activeQueue, GlobalScoring *globalScoring,
                                ScoringPerVolume *scoringPerVolume)
{
  int activeSize = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*active)[i];
    Track &currentTrack = gammas[slot];

    ComputePhysicsStepLimit(currentTrack);

    G4HepEmTrack* theTrack = currentTrack.gammaTrack.GetTrack();
    int winnerProcessIndex = theTrack->GetWinnerProcessIndex();

    bool inWorld = ComputeGeometryStepAndPropagate(currentTrack);
    atomicAdd(&globalScoring->neutralSteps, 1);

    if (theTrack->GetOnBoundary()) {
      // For now, just count that we hit something.
      atomicAdd(&globalScoring->hits, 1);

      // Kill the particle if it left the world.
      if (inWorld)
        activeQueue->push_back(slot);

      continue;
    }

    // No discrete process, move on.
    if (winnerProcessIndex < 0) {
      activeQueue->push_back(slot);
      continue;
    }

    // Reset number of interaction left for the winner discrete process.
    // (Will be resampled in the next iteration.)
    theTrack->SetNumIALeft(-1, theTrack->GetWinnerProcessIndex());

    // Perform the discrete interaction.
    switch (winnerProcessIndex) {
    case 0: {
      if (!GammaConversion(currentTrack, secondaries, globalScoring, scoringPerVolume))
        activeQueue->push_back(slot);
      break;
    }
    case 1: {
      if (!ComptonScattering(currentTrack, secondaries, globalScoring, scoringPerVolume))
        activeQueue->push_back(slot);
      break;
    }
    case 2: {
      PhotoElectricEffect(currentTrack, secondaries, globalScoring, scoringPerVolume);
      break;
    }
    }
  }
}
