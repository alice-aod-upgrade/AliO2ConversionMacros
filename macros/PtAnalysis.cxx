/// \file PtSpectrum.Cxx
/// \brief A demonstration of how to use the new analysis framework
///
/// Contains three different tasks, all computing the pt_efficieny. One
/// constructs ESD events and follows the old syntax. Two which run over raw
/// files, of which one is vectorized. Tasks are run in parallel.
///
/// \since 2017-03-09
/// \author R.G.A. Deckers
/// \copyright
///  This program is free software; you can redistribute it and/or
/// modify it under the terms of the GNU General Public License as
/// published by the Free Software Foundation; either version 3 of
/// the License, or (at your option) any later version.
/// This program is distributed in the hope that it will be useful, but
/// WITHOUT ANY WARRANTY; without even the implied warranty of
/// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
/// General Public License for more details at
/// https://www.gnu.org/copyleft/gpl.html

#include "immintrin.h"
#include <Entities/Particle.h>
#include <Entities/Track.h>
#include <Entities/Vertex.h>
#include <O2AnalysisManager.h>
#include <O2ESDAnalysisTask.h>
#include <TFile.h>
#include <TH1F.h>
#include <TThread.h>

// What type of track and vertex we want to use for our events.
using Track_t = ecs::Track<ecs::track::Pt, ecs::track::mc::MonteCarloIndex>;
using Mc_t = ecs::Particle<ecs::particle::Px, ecs::particle::Py>;
using Vertex_t = ecs::Vertex<>;

// our analysis object. By deriving from O2ESDAnalysisTask the framework will
// build events for us from datastreams using the same mapping of vertex->tracks
// that was used in the original ESD files.
class PtAnalysis : public O2ESDAnalysisTask<Vertex_t, Track_t, Mc_t> {
public:
  virtual ~PtAnalysis(){};
  // Gets called once.
  virtual void UserInit() {
    mHistogram = TH1F("ESD", "Pt Efficieny, Flat, Vectorized", 600, -0.1, 3);
  }
  // Gets called once per event.
  virtual void UserExec() {
    // get the current event as before.
    auto event = getEvent();
    // std::cout << "This event contains " << getEvent().getNumberOfTracks()
    //           << " tracks\n";
    for (int i = 0; i < event.getNumberOfTracks(); i++) {
      auto track = event.getTrack(i);
      double McLabel = track.mcLabel();
      auto mcTrack = event.getParticle(McLabel);
      mHistogram.Fill(track.pt() / mcTrack.pt());
    }
  }
  virtual void finish() { mHistogram.Write(); }

protected:
  // protected stuff goes here

private:
  TH1F mHistogram;
};

// our analysis object. By deriving from O2AnalysisTask the framework will
// not build any events but call UserExec for each input file.
class PtAnalysisFlat : public O2AnalysisTask {
public:
  // Gets called once.
  virtual void UserInit() {
    mHistogram = TH1F("flat", "Pt Efficieny, Flat, Vectorized", 600, -0.1, 3);
  }
  // Gets called once per event, which is a single file for this type.
  virtual void UserExec() {
    ecs::EntityCollection<Track_t> tracks(*(this->getHandler()));
    ecs::EntityCollection<Mc_t> particles(*(this->getHandler()));
    // std::cout << "This event contains " << getEvent().getNumberOfTracks()
    //           << " tracks\n";
    for (int i = 0; i < tracks.size(); i++) {
      auto track = tracks[i];
      auto McLabel = track.mcLabel();
      auto mcTrack = particles[McLabel];
      mHistogram.Fill(track.pt() / mcTrack.pt());
    }
  }
  virtual void finish() { mHistogram.Write(); }

protected:
  // protected stuff goes here

private:
  TH1F mHistogram;
};

// our analysis object. By deriving from O2AnalysisTask the framework will
// not build any events but call UserExec for each input file.
// NOTE: Requires avx2!
class PtAnalysisFlatVectorized : public O2AnalysisTask {
public:
  PtAnalysisFlatVectorized(int number = 0) : mNumber(number) {}
  ~PtAnalysisFlatVectorized() {}
  // Gets called once.
  virtual void UserInit() {
    mHistogram = TH1F(TString::Format("vectorized, thread %d", mNumber),
                      "Pt Efficieny, Flat, Vectorized", 600, -0.1, 3);
  }
  // Gets called once per event, which is a single file for this type.
  virtual void UserExec() {
    ecs::EntityCollection<Track_t> tracks(*(this->getHandler()));
    ecs::EntityCollection<Mc_t> particles(*(this->getHandler()));
    auto track_indices = tracks.get<ecs::track::mc::MonteCarloIndex>();
    auto track_px = tracks.get<ecs::track::Px>();
    auto track_py = tracks.get<ecs::track::Py>();

    auto particle_px = particles.get<ecs::particle::Px>();
    auto particle_py = particles.get<ecs::particle::Py>();

    // // Define a vector type of 8 floats.
    typedef float v8f __attribute__((vector_size(32)));
    int i;
    for (i = 0; i < tracks.size() - 8; i += 8) {
      //
      auto indices = _mm256_loadu_si256((__m256i const *)(track_indices + i));
      //v8f ppy = _mm256_i32gather_ps((float *)particle_py, indices, 4); //replace these lines so that the code compiles
      //v8f ppx = _mm256_i32gather_ps((float *)particle_px, indices, 4); //replace these lines so that the code compiles
      v8f ppy = _mm256_loadu_ps((float *)particle_py);
      v8f ppx = _mm256_loadu_ps((float *)particle_px);
      v8f ppt2 = (ppy * ppy + ppx * ppx);
      //
      v8f tpx = _mm256_loadu_ps((float *)track_px + i);
      v8f tpy = _mm256_loadu_ps((float *)track_py + i);
      v8f tpt2 = (tpx * tpx + tpy * tpy);
      //
      v8f pt_efficieny = _mm256_sqrt_ps(tpt2 / ppt2);
      for (unsigned u = 0; u < 8; u++) {
        mHistogram.Fill(pt_efficieny[u]);
      }
    }
    // handle the remaining tracks in a scalar fashion (max 7)
    for (; i < tracks.size(); i++) {
      auto tpt2 = track_px[i] * track_px[i] + track_py[i] * track_py[i];
      auto label = track_indices[i];
      auto ppt2 = particle_px[label] * particle_px[label] +
                  particle_py[label] * particle_py[label];
      mHistogram.Fill(sqrt(tpt2 / ppt2));
    }
  }
  virtual void finish() {
    // Don't clog the output with duplicates.
    if (mNumber == 0) {
      mHistogram.Write();
    }
  }

protected:
  // protected stuff goes here

private:
  TH1F mHistogram;
  int mNumber;
};

int PtSpectrum(const char **files, int fileCount) {

  // create the analysis manager
  O2AnalysisManager mgr;
  for (int i = 0; i < fileCount; i++) {
    mgr.addFile(files[i]);
  }

  for (int i = 0; i < 1024; i++) {
    mgr.createNewTask<PtAnalysisFlatVectorized>(i);
  }
  // Note the '&'! this has to be a reference otherwise we create a copy of
  // the newly created task and it will not be updated.
  auto &task_flat = mgr.createNewTask<PtAnalysisFlat>();
  auto &task_esd = mgr.createNewTask<PtAnalysis>();

  // open a file to put the results in.
  auto file = TFile::Open("AnalysisResult.root", "RECREATE");
  mgr.startAnalysis();
  file->Close();
  return 0;
}

#ifndef __CINT__
int main(int argc, const char **argv) { return PtSpectrum(argv + 1, argc - 1); }
#endif
