# SPDX-License-Identifier: GPL-3.0-only
"""Build a native reference from separately obtained, hash-checked sources.

Generated C++ retains upstream notices and stays outside the Python package.
Requires Python 3.10+ and a C++17 compiler, or Zig with --zig.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

REVISION = "cf7368216c73c43181e057fa33b479c4e0c86df0"
HEADER = r'''#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
namespace native_std {
using ::exp; using ::pow; using ::log; using ::sqrt;
using ::printf; using ::fprintf; using ::exit; using ::atof;
inline double abs(double x){return ::fabs(x);}
inline double min(double x,double y){return x<y?x:y;}
inline double max(double x,double y){return x>y?x:y;}
struct string_view {
 const char* p;
 constexpr string_view(const char* x):p(x){}
 constexpr const char* data()const{return p;}
 constexpr bool empty()const{return p[0]==0;}
 constexpr size_t size()const{size_t n=0;while(p[n])++n;return n;}
};
template<class T,size_t N>struct array {T v[N]; T& operator[](size_t i){return v[i];} const T& operator[](size_t i)const{return v[i];}};
}
using Real64=double;
using std::min; using std::max;
namespace Constant { constexpr double Pi=3.14159265358979323846, Kelvin=273.15, TriplePointOfWaterTempKelvin=273.16; }
namespace HVAC { enum class FanOp {Continuous, Cycling}; }
namespace WaterCoils { constexpr double MinWaterMassFlowFrac=1e-6, MinAirMassFlow=1e-6, PolyConvgTol=1e-5; constexpr int MaxOrderedPairs=60,MaxPolynomOrder=4; }
template<class T>struct Array1D {T v[128]{}; Array1D(int=128){}; T& operator()(int i){return v[i];} const T& operator()(int i)const{return v[i];} void operator=(T x){for(auto& a:v)a=x;}};
template<class T>struct Array2 {T v[128][16]{}; T& operator()(int i,int j){return v[i][j];} const T& operator()(int i,int j)const{return v[i][j];} void operator=(T x){for(auto& a:v)for(auto& b:a)b=x;}};
inline double pow_2(double x){return x*x;}
struct Schedule {double getCurrentVal(){return 1.;}};
struct Coeff { std::array<double,5> v{.98,-.3,.02,0.,0.}; double operator()(int i)const{return v[i-1];} };
struct Coil {
Schedule schedule; Schedule* availSched=&schedule;
struct {int loopNum=1;} WaterPlantLoc;
Coeff DryFinEfficncyCoef;
struct Label {const char* value; const char* operator+(const char* suffix)const{return suffix;}}; Label Name{"public synthetic coil"};
@FIELDS@
};
struct WaterData {Array2<double> OrderedPair,OrdPairSum,OrdPairSumMatrix; Coil coil; Coil& WaterCoil(int){return coil;} int DesignCalc=1; bool flag=false; bool& CoilWarningOnceFlag(int){return flag;} int errors=0; int& WaterTempCoolCoilErrs(int){return errors;} int& PartWetCoolCoilErrs(int){return errors;} };
struct EnergyPlusData;
struct Fluid {double getDensity(EnergyPlusData&,double,std::string_view){return 998.2;} double getSpecificHeat(EnergyPlusData&,double,std::string_view){return 4180.;}};
struct Plant {Fluid fluid; struct Loop {Fluid* glycol;}; Loop PlantLoop(int){return {&fluid};}};
struct Env {double OutBaroPress=101325.;};
struct Global {bool WarmupFlag=false;};
struct EnergyPlusData {WaterData w; Plant p; Env e; Global g; WaterData* dataWaterCoils=&w; Plant* dataPlnt=&p; Env* dataEnvrn=&e; Global* dataGlobal=&g;};
template<class...Args>const char* format(std::string_view s,Args...){return s.data();}
template<class...Args>void ShowWarningError(Args...){ }
template<class...Args>void ShowContinueError(Args...){ }
template<class...Args>void ShowSevereError(Args...){ }
template<class...Args>void ShowFatalError(Args...){std::exit(2);}
template<class...Args>void ShowRecurringWarningErrorAtEnd(EnergyPlusData& state,std::string_view s,Args...){state.w.errors++; std::fprintf(stderr,"%.*s\n",int(s.size()),s.data());}
double F6(double x,double a,double b,double c,double d,double e,double f){return a+x*(b+x*(c+x*(d+x*(e+x*f))));}
double F7(double x,double a,double b,double c,double d,double e,double f,double g){return a+x*(b+x*(c+x*(d+x*(e+x*(f+x*g)))));}
double PsyHFnTdbW(double t,double w){return 1004.84*t+max(w,1.e-5)*(2500940.+1858.95*t);}
double PsyCpAirFnW(double w){return 1004.84+max(w,1.e-5)*1858.95;}
double PsyRhoAirFnPbTdbW(EnergyPlusData&,double p,double t,double w,std::string_view){return p/(287.042*(t+273.15)*(1.+1.607858*w));}
@PSAT@
double PsyWFnTdbRhPb(EnergyPlusData& s,double t,double rh,double p,std::string_view who){double pv=rh*PsyPsatFnTemp(s,t,who);return max(1.e-5,.62198*pv/max(p-pv,1000.));}
double PsyHFnTdbRhPb(EnergyPlusData& s,double t,double rh,double p,std::string_view who){return PsyHFnTdbW(t,PsyWFnTdbRhPb(s,t,rh,p,who));}
// Only the saturated Tdb=Twb branch is used by PsyTsatFnHPb.
double PsyWFnTdbTwbPb(EnergyPlusData& s,double t,double twb,double p,std::string_view who){if(t!=twb)std::exit(3);return PsyWFnTdbRhPb(s,t,1.,p,who);}
double PsyTdpFnWPb(EnergyPlusData& s,double w,double p,std::string_view who){double pv=p*w/(.62198+w),lo=-99.,hi=99.;for(int i=0;i<70;i++){double mid=(lo+hi)/2.;if(PsyPsatFnTemp(s,mid,who)>pv)hi=mid;else lo=mid;}return (lo+hi)/2.;}
double PsyWFnTdbH(EnergyPlusData&,double t,double h,std::string_view){return max(1.e-5,(h-1004.84*t)/(2500940.+1858.95*t));}
@TSAT@
'''

MAIN = r'''int main(int argc,char** argv){
 EnergyPlusData state; auto& c=state.w.coil;
 c.InletAirTemp=27.; c.InletWaterTemp=7.; c.InletAirHumRat=argc>1?std::atof(argv[1]):.01;
 c.InletAirMassFlowRate=.5; c.InletWaterMassFlowRate=.4; c.MaxWaterMassFlowRate=.4;
 c.InletAirEnthalpy=PsyHFnTdbW(c.InletAirTemp,c.InletAirHumRat);
 c.FinSurfArea=18.; c.TotCoilOutsideSurfArea=20.; c.TotTubeInsideArea=2.; c.MinAirFlowArea=.4;
 c.TubeInsideDiam=.01; c.TubeOutsideDiam=.012; c.NumOfTubesPerRow=10.; c.CoilEffectiveInsideDiam=.012;
 c.TubeThermConductivity=400.; c.FinThermConductivity=200.; c.FinThickness=.00015; c.EffectiveFinDiam=.032;
 c.GeometryCoef1=.15; c.GeometryCoef2=-.3;
 if(argc>2){
  Array1D<double> coefficients;
  CalcDryFinEffCoef(state,c.TubeOutsideDiam/c.EffectiveFinDiam,coefficients);
  for(int i=1;i<=5;i++) c.DryFinEfficncyCoef.v[i-1]=coefficients(i);
  c.GeometryCoef1=.159*std::pow(c.FinThickness/c.CoilEffectiveInsideDiam,-.065)*std::pow(c.FinThickness/.01,.141);
  c.GeometryCoef2=-.323*std::pow(.001/.01,.049)*std::pow(c.EffectiveFinDiam/.03,.549)*std::pow(c.FinThickness/.001,-.028);
 }
 c.SatEnthlCurveSlope=c.EnthVsTempCurveAppxSlope=3.3867; c.SatEnthlCurveConstCoef=c.EnthVsTempCurveConst=-10.57;

 std::printf("{\"W_in\":%.16g",c.InletAirHumRat);
 if(argc>2) {
  std::printf(",\"geometry_c1\":%.16g,\"geometry_c2\":%.16g,\"dry_fin_coefficients\":[",c.GeometryCoef1,c.GeometryCoef2);
  for(int i=1;i<=5;i++)std::printf("%s%.16g",i==1?"":",",c.DryFinEfficncyCoef(i));
  std::printf("]");
 }
 CalcDetailFlatFinCoolingCoil(state,1,1,HVAC::FanOp::Continuous,1.);
 std::printf(",\"Q_W\":%.16g,\"air_out_C\":%.16g,\"water_out_C\":%.16g,\"W_out\":%.16g,\"wet_fraction\":%.16g,\"condensate_kg_s\":%.16g,\"warnings\":%d}\n",c.TotWaterCoolingCoilRate,c.OutletAirTemp,c.OutletWaterTemp,c.OutletAirHumRat,c.SurfAreaWetFraction,.5*(c.InletAirHumRat-c.OutletAirHumRat),state.w.errors);
}
'''

INSTRUMENTATION = r'''        std::printf(",\"surface_cold_C\":%.16g,\"surface_hot_trial_C\":%.16g",InCoilSurfTemp,OutCoilSurfTemp);
        if (waterCoil.SurfAreaWetFraction < 1.) std::printf(",\"interface_water_C\":%.16g,\"interface_surface_C\":%.16g",WetDryInterfcWaterTemp,WetDryInterfcSurfTemp);
        std::printf(",\"interface_air_C\":%.16g",AirWetDryInterfcTemp);
'''


def generate(source_dir: Path) -> str:
    manifest_path = Path(__file__).resolve().parents[1] / "docs/references/elmahdy_mitalas_source.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {f["path"]: f["sha256"] for f in manifest["files"]}
    def read(relative):
        path = source_dir / relative
        if not path.exists():
            path = source_dir / relative.replace("/", "__")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected[relative]:
            raise ValueError(f"Pinned source hash mismatch: {relative}")
        return content.decode("utf-8")
    cc = read("src/EnergyPlus/WaterCoils.cc")
    psy = read("src/EnergyPlus/Psychrometrics.cc")
    read("LICENSE.txt")
    coil = cc[cc.index("void CalcDetailFlatFinCoolingCoil("):cc.index("void CoolingCoil(")]
    special = {"availSched", "WaterPlantLoc", "DryFinEfficncyCoef", "Name"}
    fields = sorted(set(re.findall(r"waterCoil\.(\w+)", coil)) - special)
    header = HEADER.replace("@FIELDS@", "\n".join("double " + f + "=0.;" for f in fields))
    def psych_body(name):
        start = psy.index("Real64 " + name + "(")
        start = psy.index("#endif", start) + len("#endif")
        end = psy.index("\n    }", start) + len("\n    }")
        return psy[start:end]
    header = header.replace("@PSAT@", "double PsyPsatFnTemp(EnergyPlusData &state,double T,std::string_view CalledFrom)\n" + psych_body("PsyPsatFnTemp_raw"))
    header = header.replace("@TSAT@", 'double PsyTsatFnHPb(EnergyPlusData &state,double H,double PB,std::string_view CalledFrom="")\n' + psych_body("PsyTsatFnHPb_raw"))
    marker = "        // Set the outlet conditions\n"
    if coil.count(marker) != 1:
        raise ValueError("Native instrumentation anchor changed")
    coil = coil.replace(marker, INSTRUMENTATION + marker)
    helpers = cc[cc.index("void CalcDryFinEffCoef("):cc.index("void CoilAreaFracIter(")]
    prototypes = "\nvoid CalcIBesselFunc(Real64,int,Real64&,int&);\nvoid CalcKBesselFunc(Real64,int,Real64&,int&);\nvoid CalcPolynomCoef(EnergyPlusData&,Array2<Real64> const&,Array1D<Real64>&);\n"
    # Replace only standard-library qualification with the small harness shim;
    # the mathematical expressions and native iteration body are unchanged.
    generated = cc[:cc.index("// C++ Headers")] + header + prototypes + helpers + coil + MAIN
    return generated.replace("std::", "native_std::")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--zig", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.output_dir / "native_reference.cc"
    source.write_text(generate(args.source_dir), encoding="utf-8")
    exe = (args.output_dir / ("native_reference.exe" if os.name == "nt" else "native_reference")).resolve()
    command = [args.compiler] + (["c++"] if args.zig else [])
    command += ["-std=c++17", "-O2", "-fno-exceptions", "-fno-rtti", "-fno-threadsafe-statics"]
    if args.zig:
        command += ["-nostdlib++"]
    command += ["-o", str(exe), str(source)]
    env = dict(os.environ, ZIG_LOCAL_CACHE_DIR=str((args.output_dir / "zig-cache").resolve()),
               ZIG_GLOBAL_CACHE_DIR=str((args.output_dir / "zig-global").resolve()))
    subprocess.run(command, check=True, env=env, timeout=180)
    def run_case(w, native_fins):
        arguments = [str(exe), str(w)] + (["upstream_fins"] if native_fins else [])
        run = subprocess.run(arguments, text=True, capture_output=True, check=True, timeout=10)
        if run.stderr:
            raise RuntimeError(run.stderr)
        case = json.loads(run.stdout)
        if case["warnings"]:
            raise RuntimeError("Native convergence warning")
        if case["wet_fraction"] == 0:
            del case["surface_cold_C"]
        if case["wet_fraction"] < 1:
            del case["surface_hot_trial_C"]
        case["input_family"] = "native_fins" if native_fins else "initialized_coefficients"
        return case
    cases = [run_case(w, native_fins) for native_fins in (False, True)
             for w in (.004, .008, .01, .016)]
    limitations = [run_case(.0067, native_fins) for native_fins in (False, True)]
    evidence = {"source_revision": REVISION, "cases": cases, "near_onset_limitations": limitations,
                "native_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "native_executable_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
                "build_command": command}
    (args.output_dir / "native_results.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
