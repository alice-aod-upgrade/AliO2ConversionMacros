/// \file runConversion.Cxx
/// \brief implementation of the runConversion macro.
/// \since 2016-11-16
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

#include "OldESDPtAnalysis.h"
#include <AliAnalysisAlien.h>
#include <AliAnalysisManager.h>
#include <AliESDInputHandler.h>
#include <AliMCEventHandler.h>
#include <AliVEventHandler.h>
#include <TAlienFile.h>
#include <TChain.h>

/// Converts existing local ESDs into O2Timeframes. returns an
/// AnalysisResult.root file
///
/// \param files A list of files to convert
/// \param fileCount the number of files in files
/// \return returns 0 on success
int runConversion(const char **files, int fileCount) {
  // create the analysis manager
  AliAnalysisManager *mgr = new AliAnalysisManager("AnalysisTaskExample");
  AliESDInputHandler *ESDH = new AliESDInputHandler();
  AliMCEventHandler *mcHandler = new AliMCEventHandler();
  mgr->SetMCtruthEventHandler(mcHandler);
  mgr->SetInputEventHandler(ESDH);

  // create an instance of your analysis task
  auto *task = new PtAnalysis("Pt Task");
  if (nullptr == task) {
    return -1;
  }
  TString fileName = AliAnalysisManager::GetCommonFileName();
  fileName += ":";
  fileName += task->GetName(); // create a subfolder in the file
  // now we create an instance of your task
  mgr->ConnectInput(task, 0, mgr->GetCommonInputContainer());
// i m p o r t a n t : return a pointer to your task
  // same for the output
  mgr->ConnectOutput(task, 1,
                     mgr->CreateContainer("",
                     TList::Class(),
                                          AliAnalysisManager::kOutputContainer,
                                          fileName));
  TChain chain("esdTree");
  for (int i = 0; i < fileCount; i++) {
    chain.Add(files[i]);
  }
  if (!mgr->InitAnalysis()) {
    return -2;
  }
  mgr->PrintStatus();
  mgr->SetUseProgressBar(1, 25);
  mgr->StartAnalysis("local", &chain);
  return 0;
}

#ifndef __CINT__
int main(int argc, const char **argv) {
  return runConversion(argv + 1, argc - 1);
}
#endif
