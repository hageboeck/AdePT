// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0
//
#include "ParticleGun.hh"
#include "ParticleGunMessenger.hh"

#include "G4PhysicalConstants.hh"
#include "G4PrimaryParticle.hh"
#include "G4Event.hh"
#include "Randomize.hh"
#include "G4SystemOfUnits.hh"
#include "G4ParticleTable.hh"

#ifdef HEPMC3_FOUND
#include "HepMC3G4AsciiReader.hh"
#endif

#include <numeric>

ParticleGun::ParticleGun() : G4ParticleGun()
{
  SetDefaultKinematic();
  // if HepMC3, create the reader
#ifdef HEPMC3_FOUND
  fHepmcAscii = std::make_unique<HepMC3G4AsciiReader>();
#endif

  fMessenger = std::make_unique<ParticleGunMessenger>(this);
}

ParticleGun::~ParticleGun()
{

}

void ParticleGun::GeneratePrimaries(G4Event *aEvent)
{
  // this function is called at the begining of event
  //

  if (fRandomizeGun) {
    if (!fInitializationDone) {
      // We only need to do this for the first run
      fInitializationDone = true;
      // Re-balance the user-provided weights if needed
      ReWeight();
      // Make sure all particles have a user-defined energy
      for (unsigned int i = 0; i < fParticleEnergies.size(); i++) {
        if (fParticleEnergies[i] < 0) {
          G4Exception("PrimaryGeneratorAction::GeneratePrimaries()", "Notification", FatalErrorInArgument,
                      ("Energy undefined for  " + fParticleList[i]->GetParticleName()).c_str());
        }
      }
      // In case the upper range for Phi or Theta was not defined, or is lower than the
      // lower range
      if (fMaxPhi < fMinPhi) {
        fMaxPhi = fMinPhi;
      }
      if (fMaxTheta < fMinTheta) {
        fMaxTheta = fMinTheta;
      }
    }
  }

  if (fUseHepMC && fHepmcAscii) {
    fHepmcAscii->GeneratePrimaryVertex(aEvent);
  } else {
    if (fRandomizeGun) {
      GenerateRandomPrimaryVertex(aEvent, fMinPhi, fMaxPhi, fMinTheta, fMaxTheta, &fParticleList, &fParticleWeights,
                                  &fParticleEnergies);
    } else {
      GeneratePrimaryVertex(aEvent);
    }
  }

  // Print the particle gun info if requested
  if (fPrintGun) Print();

  PrintPrimaries(aEvent);
}

void ParticleGun::GenerateRandomPrimaryVertex(G4Event *aEvent, G4double aMinPhi, G4double aMaxPhi, G4double aMinTheta,
                                              G4double aMaxTheta, std::vector<G4ParticleDefinition *> *aParticleList,
                                              std::vector<float> *aParticleWeights,
                                              std::vector<float> *aParticleEnergies)
{
  // Choose a particle from the list
  float choice = G4UniformRand();
  float weight = 0;

  for (unsigned int i = 0; i < aParticleList->size(); i++) {
    weight += (*aParticleWeights)[i];
    if (weight > choice) {
      SetParticleDefinition((*aParticleList)[i]);
      SetParticleEnergy((*aParticleEnergies)[i]);
      break;
    }
  }

  // Create a new vertex
  //
  auto *vertex = new G4PrimaryVertex(particle_position, particle_time);

  // Create new primaries and set them to the vertex
  //
  G4double mass = particle_definition->GetPDGMass();
  for (G4int i = 0; i < NumberOfParticlesToBeGenerated; ++i) {
    auto *particle = new G4PrimaryParticle(particle_definition);

    // Choose a random direction in the selected ranges, with an isotropic distribution
    G4double phi                = (aMaxPhi - aMinPhi) * G4UniformRand() + aMinPhi;
    G4double theta              = acos((cos(aMaxTheta) - cos(aMinTheta)) * G4UniformRand() + cos(aMinTheta));
    G4double x                  = cos(phi) * sin(theta);
    G4double y                  = sin(phi) * sin(theta);
    G4double z                  = cos(theta);
    particle_momentum_direction = G4ThreeVector(x, y, z);

    particle->SetKineticEnergy(particle_energy);
    particle->SetMass(mass);
    particle->SetMomentumDirection(particle_momentum_direction);
    particle->SetCharge(particle_charge);
    particle->SetPolarization(particle_polarization.x(), particle_polarization.y(), particle_polarization.z());
    vertex->SetPrimary(particle);

    // Choose a new particle from the list for the next iteration
    choice = G4UniformRand();
    weight = 0;
    for (unsigned int i = 0; i < aParticleList->size(); i++) {
      weight += (*aParticleWeights)[i];
      if (weight > choice) {
        SetParticleDefinition((*aParticleList)[i]);
        SetParticleEnergy((*aParticleEnergies)[i]);
        break;
      }
    }
  }
  aEvent->AddPrimaryVertex(vertex);
}

void ParticleGun::SetDefaultKinematic()
{
  G4ParticleTable *particleTable = G4ParticleTable::GetParticleTable();
  G4String particleName;
  G4ParticleDefinition *particle = particleTable->FindParticle(particleName = "e-");
  SetParticleDefinition(particle);
  SetParticleMomentumDirection(G4ThreeVector(1., 1., 1.));
  SetParticleEnergy(1. * GeV);
  G4double position = 0.0;
  SetParticlePosition(G4ThreeVector(position, 0. * cm, 0. * cm));
}

void ParticleGun::AddParticle(G4ParticleDefinition *val, float weight, double energy)
{
  fParticleList.push_back(val);
  fParticleWeights.push_back(weight);
  fParticleEnergies.push_back(energy);
}

void ParticleGun::ReWeight()
{
  double userDefinedSum = 0;
  double numNotDefined  = 0;
  for (float i : fParticleWeights)
    i >= 0 ? userDefinedSum += i : numNotDefined += 1;

  if (userDefinedSum < 1 && numNotDefined == 0) {
    // If the user-provided weights do not sum up to 1 and there are no particles left to
    // distribute the remaining weight, re-balance their weights
    for (unsigned int i = 0; i < fParticleWeights.size(); i++) {
      fParticleWeights[i] = fParticleWeights[i] / userDefinedSum;
      G4Exception("PrimaryGeneratorAction::ReWeight()", "Notification", JustWarning,
                  ("Sum of user-defined weights is <1, new weight for " + fParticleList[i]->GetParticleName() + " = " +
                   std::to_string(fParticleWeights[i]))
                      .c_str());
    }
  } else {
    for (unsigned int i = 0; i < fParticleWeights.size(); i++) {
      double originalWeight = fParticleWeights[i];
      // Particles with no user-defined weight have weight -1
      if (originalWeight >= 0) {
        // For particles with user-defined weight, re-balance only if the sum is higher than 1
        if (userDefinedSum <= 1) {
          fParticleWeights[i] = originalWeight;
        } else {
          fParticleWeights[i] = originalWeight / userDefinedSum;
          G4Exception("PrimaryGeneratorAction::ReWeight()", "Notification", JustWarning,
                      ("Sum of user-defined weights is >1, new weight for " + fParticleList[i]->GetParticleName() +
                       " = " + std::to_string(fParticleWeights[i]))
                          .c_str());
        }
      } else if (userDefinedSum < 1) {
        // For particles with no user-defined weight, distribute the remaining weight
        fParticleWeights[i] = (1 - userDefinedSum) / numNotDefined;
      } else {
        // If the sum of user-defined weights is greater or equal to 1 there's nothing left to distribute,
        // the probability for the remaining particles will be 0
        fParticleWeights[i] = 0;
      }
    }
  }
}

void ParticleGun::Print()
{
  if (!fRandomizeGun) {
    G4cout << "=== Gun shooting " << GetParticleDefinition()->GetParticleName() << " with energy "
           << GetParticleEnergy() / GeV << "[GeV] from: " << GetParticlePosition() / mm
           << " [mm] along direction: " << GetParticleMomentumDirection() << G4endl;
  } else {
    for (unsigned int i = 0; i < fParticleList.size(); i++) {
      G4cout << "=== Gun shooting " << fParticleEnergies[i] / GeV << "[GeV] " << fParticleList[i]->GetParticleName()
             << " with probability " << fParticleWeights[i] * 100 << "%" << G4endl;
    }
    G4cout << "=== Gun shooting from: " << GetParticlePosition() / mm << " [mm]" << G4endl;
    G4cout << "=== Gun shooting in ranges: " << G4endl;
    G4cout << "Phi: [" << fMinPhi << ", " << fMaxPhi << "] (rad)" << G4endl;
    G4cout << "Theta: [" << fMinTheta << ", " << fMaxTheta << "] (rad)" << G4endl;
  }
}

void ParticleGun::PrintPrimaries(G4Event *aEvent) const
{
  std::map<G4String, G4int> aParticleCounts             = {};
  std::map<G4String, G4double> aParticleAverageEnergies = {};

  for (int i = 0; i < aEvent->GetPrimaryVertex()->GetNumberOfParticle(); i++) {
    G4String aParticleName   = aEvent->GetPrimaryVertex()->GetPrimary(i)->GetParticleDefinition()->GetParticleName();
    G4double aParticleEnergy = aEvent->GetPrimaryVertex()->GetPrimary(i)->GetKineticEnergy();
    if (!aParticleCounts.count(aParticleName)) {
      aParticleCounts[aParticleName]          = 0;
      aParticleAverageEnergies[aParticleName] = 0;
    }
    aParticleCounts[aParticleName] += 1;
    aParticleAverageEnergies[aParticleName] += aParticleEnergy;
  }

  for (auto pd : aParticleCounts) {
    G4cout << pd.first << ": " << pd.second << ", " << aParticleAverageEnergies[pd.first] / pd.second << G4endl;
  }
}