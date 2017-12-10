/// \file ConversionAnalysis.h
/// \author R.G.A. Deckers
/// \brief Definition of the ConversionAnalysis class.
///
/// See implementation file for copyright details.

#ifndef CONVERSIONANALYSIS_H
#define CONVERSIONANALYSIS_H

// Include the base classes
#include <AliAnalysisTaskSE.h>
#include <ESDEventConverter.h>
#include <TRandom2.h>

class ConversionAnalysis : public AliAnalysisTaskSE {
public:
  /// Default constructor
  ConversionAnalysis();
  /// Named constructor
  ConversionAnalysis(const char *name);
  /// Destructor
  ~ConversionAnalysis();
  // intialization
  virtual void UserCreateOutputObjects();
  // per event
  virtual void UserExec(Option_t *option);
  // Cleanup
  virtual void Terminate(Option_t *option);
  void usingMcData(bool setTo) { mConverter.useMcInfo(setTo); }

protected:
  // protected stuff goes here

private:
  /// copy constructor prohibited
  ConversionAnalysis(const ConversionAnalysis &);
  /// assignment operator prohibited
  ConversionAnalysis &operator=(const ConversionAnalysis &);
  TRandom2 mRng;
  double mCurrentTimestamp = 0.0f;
  unsigned mEventsOnQueue = 0;
  o2::ESDEventConverter mConverter;
  // root specific
  ClassDef(ConversionAnalysis, 1);
};

#endif
