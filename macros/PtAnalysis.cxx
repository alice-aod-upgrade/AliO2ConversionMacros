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

#include <chrono>
using namespace std;
using namespace chrono;
#include "immintrin.h"
#include <Entities/Particle.h>
#include <Entities/Track.h>
#include <Entities/Vertex.h>
#include <O2AnalysisManager.h>
#include <O2ESDAnalysisTask.h>
#include <expression_templates/Histogram.hpp>
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
  PtAnalysis(int number = 0) : mNumber(number) {}
  ~PtAnalysis() {}
  // Gets called once.
  virtual void UserInit() {
    mHistogram = TH1F(TString::Format("thread %d", mNumber), "Pt Efficieny ",
                      600, -0.1, 3);
  }
  // Gets called once per event.
  virtual void UserExec() {
    // get the current event as before.
    auto event = getEvent();
    // std::cout << "This event contains " << getEvent().getNumberOfTracks()
    //           << " tracks\n";
    for (int i = 0; i < event.getNumberOfTracks(); i++) {
      auto track = event.getTrack(i);
      int McLabel = track.mcLabel();
      auto mcTrack = event.getParticle(McLabel);
      mHistogram.Fill(track.pt() / mcTrack.pt());
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

// our analysis object. By deriving from O2AnalysisTask the framework will
// not build any events but call UserExec for each input file.
class PtAnalysisFlat : public O2AnalysisTask {
public:
  PtAnalysisFlat(int number = 0) : mNumber(number) {}
  ~PtAnalysisFlat() {}
  // Gets called once.
  virtual void UserInit() {
    mHistogram = TH1F(TString::Format("flat, thread %d", mNumber),
                      "Pt Efficieny, Flat", 600, -0.1, 3);
  }
  // Gets called once per event, which is a single file for this type.
  virtual void UserExec() {
    high_resolution_clock::time_point begin, end;
    begin =  high_resolution_clock::now();
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
    end = high_resolution_clock::now();
    double duration = duration_cast<microseconds>( end - begin ).count();
    std::cout << "Finished normal in \t" << duration << "ms"<<std::endl;
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

class PtAnalysisFlatAutovec : public O2AnalysisTask {
public:
  PtAnalysisFlatAutovec(int number = 0) : mNumber(number) {}
  ~PtAnalysisFlatAutovec() {}
  // Gets called once.
  virtual void UserInit() {
    mHistogram = Histogram(600, -0.1, 3);
  }
  // Gets called once per event, which is a single file for this type.
  virtual void UserExec() {
    high_resolution_clock::time_point begin, end;
    begin =  high_resolution_clock::now();

    //fetch our tracks (initializes the storage backend for the handler)
    ecs::EntityCollection<Track_t> tracks(*(this->getHandler()));
    //fetch our MonteCarlo particles.
    ecs::EntityCollection<Mc_t> particles(*(this->getHandler()));

    //grab the indices mapping tracks to particles, returns an Array/Valaray-like
    //object.
    auto track_indices = tracks.mcLabel();
    //grab the track and particle Pt. Returns an automatically deduced 'Expression type'.
    // this could be another Valaray or it could be (and currently is) an expression
    // saying "the sqrt of (get<Px>^2 plus get<Py>^2)".
    //
    // The techincal return type would be:
    //  ExpressionMap<Sqrt,
    //   ExpressionMap<Sum,   <-- argument to sqrt
    //     ExpressionMap<Mul, <-- first arg to Sum
    //      Slice<float>,     <-- first arg to Mul = Px
    //      Slice<float>      <-- Second arg to Mul = Px
    //     >,
    //     ExpressionMap<Mul, <-- Second Arg to sum
    //      Slice<float>,     <-- First arg to Mul = Py
    //      Slice<float>      <-- First arg to Mul = Py
    //     >
    //    >
    //  >
    //
    // No data is read or computatin performed until the resulting expression gets indexed
    // with operator[] or get_vec(). If get_vec is used it computes the resulting expression using
    // SIMD expressions (SSE/AVX)
    auto track_pt = tracks.pt();
    auto particle_pt = particles.pt();

    // gather returns another expression, telling the compiler that
    // particle_pt[i] should be read as particle_pt[track_indices[i]].
    // this again supports vectorization using gather instruction where available
    // where they are not, it performs a regular scalar read.
    //
    // the division operator returns yet another expression and thus
    // result is an expression itself.
    auto result = track_pt/(particle_pt.gather(track_indices));
    // up to this point, no computation has been performed.
    // We can now use this expression to fill a histogram
    // For the best result, we fill the histogram using our own class which
    // evaluates 'result' in vector form and fills in parallel.
    // it does so by using openMP for parralizing the code and the
    // semi-auto vectorization operations supported by our expression system.
    mHistogram.Fill(result);

    // alternatively, we could use the map_expr funtion to call
    // a fill operation on each element. This still evaluates the expr`
    // in vector form but is not multithreaded because of the possible
    // side effects of the function argument. Works with root histograms.
    // map_expr(result, [this](double x){mHistogram.Fill(x);});

    // Or, we can cast the result to a Varray object. A Varrary is an expression object that also
    // stores it's own data in an array. it is like the Valaray object in the std library but
    // it is made to work with the general expression framework presented here.
    // Varray's can be constructed from expressions, when doing so they will evaluate the supplied expression
    // using vectorized instructions and parallelization.
    //
    // This uses extra memory for storing the array of the result before reading it into a histogram
    // Works with ROOT histograms.
    // Varray<float> result = track_pt/(particle_pt.gather(track_indices));
    // for(int i = 0; i < result.size(); i++){
    //   mHistogram.Fill(result[i]);
    // }
    end = high_resolution_clock::now();
    double duration = duration_cast<microseconds>( end - begin ).count();
    std::cout << "Finished autovec in \t" << duration << "ms"<<std::endl;
  }
  virtual void finish() {
    // Don't clog the output with duplicates.
    if (mNumber == 0) {
      //Create and write a root histogram from our custom histogram type.
      mHistogram.createTH1I(TString::Format("flat auto, thread %d", mNumber),
                        "flat auto").Write();
      // mHistogram.Write();
    }
  }

protected:
  // protected stuff goes here

private:
  Histogram mHistogram;
  int mNumber;
};

class PtAnalysisAutovec : public O2ESDAnalysisTask<Vertex_t, Track_t, Mc_t> {
public:
  PtAnalysisAutovec(int number = 0) : mNumber(number) {}
  ~PtAnalysisAutovec() {}
  // Gets called once.
  virtual void UserInit() {
    mHistogram = Histogram(600, -0.1, 3);
  }
  // Gets called once per event, which is a single file for this type.
  virtual void UserExec() {
    auto event = getEvent();

    auto tracks = event.getTracks(); //returns a Slice that's  subslice
    auto particles = event.getParticles(); //returns all particles from the file

    mHistogram.Fill(tracks.pt()/(particles.pt().gather(tracks.mcLabel())));
  }
  virtual void finish() {
    // Don't clog the output with duplicates.
    if (mNumber == 0) {
      //Create and write a root histogram from our custom histogram type.
      mHistogram.createTH1I(TString::Format("auto, thread %d", mNumber),
                        "auto").Write();
      // mHistogram.Write();
    }
  }

protected:
  // protected stuff goes here

private:
  Histogram mHistogram;
  int mNumber;
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
    high_resolution_clock::time_point begin, end;
    begin =  high_resolution_clock::now();
    ecs::EntityCollection<Track_t> tracks(*(this->getHandler()));
    ecs::EntityCollection<Mc_t> particles(*(this->getHandler()));
    auto track_indices = tracks.get<ecs::track::mc::MonteCarloIndex>().data();
    // ideal:
    //  hist(tracks.pt()/(particles.pt()[track_indices]))
    // auto vectorized/multithreaded.
    // tracks.pt() returns an expression that is sqrt(px*px+py*py)
    // same for particles.pt()
    // expression indexed by a varr<int> gives a gather expression.
    // divide the results for another expression.
    // hist then takes an expression and evaluates it in an optimized fashion.
    // and produces a histgram.
    // the end result should be a single loop that uses vectorized instructions
    // where possible and multithreads automatically.
    //  The resulting histogram can then be added to a ROOT hist in O(nBins).
    // The bulk compute will only be limited by the memory throughput and in some
    // cases the compute.

    auto track_px = tracks.get<ecs::track::Px>().data();
    auto track_py = tracks.get<ecs::track::Py>().data();

    auto particle_px = particles.get<ecs::particle::Px>().data();
    auto particle_py = particles.get<ecs::particle::Py>().data();

    // // Define a vector type of 8 floats.
    typedef float v8f __attribute__((vector_size(32)));
    int i;
    for (i = 0; i < tracks.size() - 8; i += 8) {
      //
      auto indices = _mm256_loadu_si256((__m256i const *)(track_indices + i));
      v8f ppy = _mm256_i32gather_ps((float *)particle_py, indices, 4);
      v8f ppx = _mm256_i32gather_ps((float *)particle_px, indices, 4);
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
    end = high_resolution_clock::now();
    double duration = duration_cast<microseconds>( end - begin ).count();
    std::cout << "Finished explicit in \t" << duration << "ms"<<std::endl;
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

  for (int i = 0; i < 64; i++) {
    mgr.createNewTask<PtAnalysisFlatVectorized>(i);
    mgr.createNewTask<PtAnalysisFlat>(i);
    mgr.createNewTask<PtAnalysis>(i);
    mgr.createNewTask<PtAnalysisFlatAutovec>(i);
    mgr.createNewTask<PtAnalysisAutovec>(i);
  }
  // Note the '&'! this has to be a reference otherwise we create a copy of
  // the newly created task and it will not be updated.
  // auto &task_flat = mgr.createNewTask<PtAnalysisFlat>();
  // auto &task_esd = mgr.createNewTask<PtAnalysis>();

  // open a file to put the results in.
  auto file = TFile::Open("AnalysisResult.root", "RECREATE");
  mgr.startAnalysis();
  file->Close();
  return 0;
}

#ifndef __CINT__
int main(int argc, const char **argv) { return PtSpectrum(argv + 1, argc - 1); }
#endif
