/// \file PtAnalysis.cxx
/// \brief implementation of the PtAnalysis class.
/// \since 2016-11-15
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
///

#include "OldESDPtAnalysis.h"
#include <AliESDEvent.h>
#include <AliLog.h>
#include <TChain.h>
#include <AliMCEvent.h>
#include <AliMCParticle.h>
#include <cassert>
// root specific
ClassImp(PtAnalysis);

// default constructor
PtAnalysis::PtAnalysis() {}
// named constructor
PtAnalysis::PtAnalysis(const char *name)
    : AliAnalysisTaskSE(name) {
  DefineInput(0, TChain::Class());
  DefineOutput (1 , TList::Class () ) ;
}
// default destructor
PtAnalysis::~PtAnalysis() {delete mOutputList;}

// User time (seconds): 390.51
// System time (seconds): 3.70
// Percent of CPU this job got: 98%
// Elapsed (wall clock) time (h:mm:ss or m:ss): 6:41.04
// Average shared text size (kbytes): 0
// Average unshared data size (kbytes): 0
// Average stack size (kbytes): 0
// Average total size (kbytes): 0
// Maximum resident set size (kbytes): 691564
// Average resident set size (kbytes): 0
// Major (requiring I/O) page faults: 0
// Minor (reclaiming a frame) page faults: 478371
// Voluntary context switches: 8478
// Involuntary context switches: 28609
// Swaps: 0
// File system inputs: 14400184
// File system outputs: 24
// Socket messages sent: 0
// Socket messages received: 0
// Signals delivered: 0
// Page size (bytes): 4096
// Exit status: 0


void PtAnalysis::UserCreateOutputObjects() {
  // create output objects
    //
    // this function is called ONCE at the start of your analysis (RUNTIME)
    // here you ceate the histograms that you want to use
    //
    // the histograms are in this case added to a tlist, this list is in the end saved
    // to an output file
    //
    mOutputList = new TList();          // this is a list which will contain all of your histograms
                                        // at the end of the analysis, the contents of this list are written
                                        // to the output file
    mOutputList->SetOwner(kTRUE);       // memory stuff: the list is owner of all objects it contains and will delete them
                                        // if requested (dont worry about this now)

    // example of a histogram
    mHistogram = new  TH1F("Old ESD",
                      "Pt Efficieny", 600, -0.1, 3);       // create your histogra
    mOutputList->Add(mHistogram);          // don't forget to add it to the list! the list will be written to file, so if you want
                                        // your histogram in the output file, add it to the list!

    PostData(1, mOutputList);           // postdata will notify the analysis manager of changes / updates to the
                                        // fOutputList object. the manager will in the end take care of writing your output to file
// so it needs to know what's in the output
}
// per event
void PtAnalysis::UserExec(Option_t *option) {
  const AliESDEvent *event = dynamic_cast<AliESDEvent *>(InputEvent());
  const auto *mcEvent = MCEvent();
  if (!event) {
    AliError(TString::Format("Failed to fetch event"));
    return;
  }
  if (!mcEvent) {
    AliError(TString::Format("Failed to fetch event"));
    return;
  }
  int numberOfTracks = event->GetNumberOfTracks();
  for (int i = 0; i < numberOfTracks; i++) {
    AliESDtrack *esdTrack = event->GetTrack(i);
    Int_t label = TMath::Abs(esdTrack->GetLabel());
    if(mcEvent->GetNumberOfTracks() < label){
      AliError(TString::Format("Out of range! %d / %d", label, mcEvent->GetNumberOfTracks()));
      return;
    }
    const AliMCParticle *mcParticle =
          (const AliMCParticle *)mcEvent->GetTrack(label);
    mHistogram->Fill(esdTrack->Pt() / mcParticle->Pt());
  }
   PostData(1, mOutputList);
}
// Cleanup
void PtAnalysis::Terminate(Option_t *option) {
}
