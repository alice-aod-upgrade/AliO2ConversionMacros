/// \file ConversionAnalysis.cxx
/// \brief implementation of the ConversionAnalysis class.
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

#include "ConversionAnalysis.h"
#include <AliESDEvent.h>
#include <AliLog.h>
#include <TChain.h>
// root specific
ClassImp(ConversionAnalysis);

// default constructor
ConversionAnalysis::ConversionAnalysis() {}
// named constructor
ConversionAnalysis::ConversionAnalysis(const char *name)
    : AliAnalysisTaskSE(name) {
  DefineInput(0, TChain::Class());
}
// default destructor
ConversionAnalysis::~ConversionAnalysis() {}

void ConversionAnalysis::UserCreateOutputObjects() {}
// per event
void ConversionAnalysis::UserExec(Option_t *option) {

  if (mCurrentTimestamp <= 20000000) {
    const AliESDEvent *event = dynamic_cast<AliESDEvent *>(InputEvent());
    if (!event) {
      AliError(TString::Format("Failed to fetch ESD event"));
      return;
    }
    // TODO: We can probably do this much faster by using math
    Double_t mu = 25.0 / 20000.0;
    double offset = 0;
    while (0 == mEventsOnQueue) {
      mEventsOnQueue = mRng.Poisson(mu);
      offset += 25;
    }
    mCurrentTimestamp += offset;
    if (mCurrentTimestamp > 20000000) {
      AliError(TString::Format("hit 20ms"));
    } else {
      mConverter.addESDEvent(mCurrentTimestamp, event, MCEvent());
    }
    mEventsOnQueue -= 1;
  }
  // AliError(TString::Format("total sim time: %f", mCurrentTimestamp));
}
// Cleanup
void ConversionAnalysis::Terminate(Option_t *option) {
  mConverter.toFile("aod.bin");
  // AliError(TString::Format("total sim time: %f", mCurrentTimestamp));
}
