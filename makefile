rwildcard=$(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) $(filter $(subst *, %,$2),$d))
directories=$(sort $(dir $(wildcard $/*)))
REPORT = @echo "\033[4;1;37m$1\033[0m"
CHK_DIR_EXISTS = test -d $1 || mkdir -p $1
NUKE = rm -r $1
COPY_DIR = cp -rv $1 $2
MV = cp -v $1 $2
LIB_SUFFIX :=.so

PROJECT_DIR :=$(dir $(realpath $(lastword $(MAKEFILE_LIST))))
OBJ_DIR := $(PROJECT_DIR)obj
BIN_DIR := $(PROJECT_DIR)bin
CLASS_DIR := $(PROJECT_DIR)classes
DICTIONARY_DIR := $(CLASS_DIR)/dictionaries
MACRO_DIR := $(PROJECT_DIR)macros
COMMON_MACRO_DIR := $(PROJECT_DIR)macros/common

CLASSES_IMPL := $(wildcard $(CLASS_DIR)/*.cxx)
CLASSES_HEADERS := $(wildcard $(CLASS_DIR)/*.h)
CLASSES_OBJ = $(patsubst $(CLASS_DIR)/%.cxx, $(OBJ_DIR)/classes/%.o,$(CLASSES_IMPL))

LINKDEFS = $(wildcard $(DICTIONARY_DIR)/*\_LinkDef.h)
DICTIONARIES = $(patsubst $(DICTIONARY_DIR)/%_LinkDef.h, $(DICTIONARY_DIR)/%_dict.cxx,$(LINKDEFS))
DICTIONARY_OBJ = $(patsubst $(DICTIONARY_DIR)/%_LinkDef.h, $(OBJ_DIR)/dictionaries/%.o,$(LINKDEFS))

MACROS := $(wildcard $(MACRO_DIR)/*.cxx)
MACROS_OBJ := $(patsubst $(MACRO_DIR)/%.cxx, $(OBJ_DIR)/macros/%.o,$(MACROS))
COMMON_MACROS := $(wildcard $(COMMON_MACRO_DIR)/*.cxx)
COMMON_MACROS_OBJ := $(patsubst $(COMMON_MACRO_DIR)/%.cxx, $(OBJ_DIR)/macros/common/%.o,$(COMMON_MACROS))
BINARIES := $(patsubst $(MACRO_DIR)/%.cxx, $(BIN_DIR)/%,$(MACROS))

ALL_OBJ := $(CLASSES_OBJ) $(COMMON_MACROS_OBJ) $(MACROS_OBJ) $(DICTIONARY_OBJ)

CPP_FLAGS := $(shell root-config --cflags) -fopenmp  -fPIC -g -fno-omit-frame-pointer -O2 -pipe -march=native -I $(CLASS_DIR) -I $(COMMON_MACRO_DIR) -I $$ALICE_ROOT/include  --std=c++14
LD_FLAGS := -fopenmp -lCling -lESD -lEG -lGeom -lMinuit -lVMC -lXMLParser -lTreePlayer -lXMLIO -lSTEERBase -lANALYSIS -lAOD -lANALYSISalice -lANALYSISaliceBase -lO2 -lboost_system -lboost_filesystem -lboost_iostreams --std=c++14 
LD_FLAGS += $(shell root-config --glibs) -L $$ALICE_ROOT/lib

.SECONDARY: $(ALL_OBJ)

binaries: $(BINARIES)

objects: $(ALL_OBJ)

$(BIN_DIR)/% : $(OBJ_DIR)/macros/%.o $(CLASSES_OBJ) $(COMMON_MACROS) $(DICTIONARY_OBJ)
	$(call REPORT,linking $@)
	$(call CHK_DIR_EXISTS, $(dir $@))
	clang++ -o "$@" $^ $(LD_FLAGS)

$(OBJ_DIR)/classes/%.o: $(CLASS_DIR)/%.cxx $(CLASS_DIR)/%.h
	$(call REPORT,Compiling $@)
	$(call CHK_DIR_EXISTS, $(dir $@))
	clang++ $(CPP_FLAGS) -o "$@" -c $<

$(OBJ_DIR)/dictionaries/%.o: $(DICTIONARY_DIR)/%_dict.cxx
	$(call REPORT,Compiling $@)
	$(call CHK_DIR_EXISTS, $(dir $@))
	clang++ $(CPP_FLAGS) -o "$@" -c $<

$(OBJ_DIR)/macros/%.o: $(MACRO_DIR)/%.cxx
	$(call REPORT,Compiling $@)
	$(call CHK_DIR_EXISTS, $(dir $@))
	clang++ $(CPP_FLAGS) -o "$@" -c $<

$(OBJ_DIR)/macros/common/%.o: $(MACRO_DIR)/common/%.cxx
	$(call REPORT,Compiling $@)
	$(call CHK_DIR_EXISTS, $(dir $@))
	clang++ $(CPP_FLAGS) -o "$@" -c $<

$(DICTIONARY_DIR)/%_dict.cxx: $(CLASS_DIR)/%.h $(DICTIONARY_DIR)/%_LinkDef.h
	$(call REPORT,Creating dictionary $@)
	rootcling -f $@ -c -I$(CLASS_DIR) -I$(COMMON_MACRO_DIR) -I$$ALICE_ROOT/include  -p $^
	$(call CHK_DIR_EXISTS, $(BIN_DIR))
	mv $(patsubst $(DICTIONARY_DIR)/%_dict.cxx, $(DICTIONARY_DIR)/%_dict_rdict.pcm,$@) $(BIN_DIR)

clean:
	$(call REPORT,Cleaning...)
	-$(NUKE) "$(OBJ_DIR)" "$(BIN_DIR)"
