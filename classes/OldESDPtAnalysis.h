/// \file PtAnalysis.h
/// \author R.G.A. Deckers
/// \brief Definition of the PtAnalysis class.
///
/// See implementation file for copyright details.

#pragma once

// Include the base classes
#include <AliAnalysisTaskSE.h>
#include <TList.h>
#include <TH1F.h>

class PtAnalysis : public AliAnalysisTaskSE {
public:
  /// Default constructor
  PtAnalysis();
  /// Named constructor
  PtAnalysis(const char *name);
  /// Destructor
  ~PtAnalysis();
  // intialization
  virtual void UserCreateOutputObjects();
  // per event
  virtual void UserExec(Option_t *option);
  // Cleanup
  virtual void Terminate(Option_t *option);

protected:
  // protected stuff goes here

private:
  /// copy constructor prohibited
  PtAnalysis(const PtAnalysis &);
  /// assignment operator prohibited
  PtAnalysis &operator=(const PtAnalysis &);
  TH1F *mHistogram;
  TList *mOutputList;
  // root specific
  ClassDef(PtAnalysis, 1);
};
