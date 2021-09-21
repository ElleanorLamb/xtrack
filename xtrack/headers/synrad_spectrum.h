#ifndef XTRACK_SYNRAD_SPECTRUM_H
#define XTRACK_SYNRAD_SPECTRUM_H

#include <math.h>

typedef struct { double x; } LocalParticle;

extern double LocalParticle_generate_random_double(LocalParticle * );

// x :    energy normalized to the critical energy
// returns function value _SynRadC   photon spectrum dn/dx
// (integral of modified 1/3 order Bessel function)
// principal: Chebyshev series see H.H.Umstaetter CERN/PS/SM/81-13 10-3-1981
// see also my LEP Note 632 of 12/1990
// converted to C++, H.Burkhardt 21-4-1996    */

/*gpufun*/
double SynRad(double x)
{ 
  double synrad = 0.;
  if(x>0. && x<800.) {	// otherwise result synrad remains 0
    if(x<6.) {
      double a,b,z;
      z=x*x/16.-2.;
      b=          .00000000000000000012;
      a=z*b  +    .00000000000000000460;
      b=z*a-b+    .00000000000000031738;
      a=z*b-a+    .00000000000002004426;
      b=z*a-b+    .00000000000111455474;
      a=z*b-a+    .00000000005407460944;
      b=z*a-b+    .00000000226722011790;
      a=z*b-a+    .00000008125130371644;
      b=z*a-b+    .00000245751373955212;
      a=z*b-a+    .00006181256113829740;
      b=z*a-b+    .00127066381953661690;
      a=z*b-a+    .02091216799114667278;
      b=z*a-b+    .26880346058164526514;
      a=z*b-a+   2.61902183794862213818;
      b=z*a-b+  18.65250896865416256398;
      a=z*b-a+  92.95232665922707542088;
      b=z*a-b+ 308.15919413131586030542;
      a=z*b-a+ 644.86979658236221700714;
      double p;
      p=.5*z*a-b+  414.56543648832546975110;
      a=          .00000000000000000004;
      b=z*a+      .00000000000000000289;
      a=z*b-a+    .00000000000000019786;
      b=z*a-b+    .00000000000001196168;
      a=z*b-a+    .00000000000063427729;
      b=z*a-b+    .00000000002923635681;
      a=z*b-a+    .00000000115951672806;
      b=z*a-b+    .00000003910314748244;
      a=z*b-a+    .00000110599584794379;
      b=z*a-b+    .00002581451439721298;
      a=z*b-a+    .00048768692916240683;
      b=z*a-b+    .00728456195503504923;
      a=z*b-a+    .08357935463720537773;
      b=z*a-b+    .71031361199218887514;
      a=z*b-a+   4.26780261265492264837;
      b=z*a-b+  17.05540785795221885751;
      a=z*b-a+  41.83903486779678800040;
      double q;
      q=.5*z*a-b+28.41787374362784178164;
      double y;
      y=pow(x,2./3.);
      synrad=(p/y-q*y-1.)*1.81379936423421784215530788143;

    } else {// 6 < x < 174

      double a,b,z;
      z=20./x-2.;
      a=      .00000000000000000001;
      b=z*a  -.00000000000000000002;
      a=z*b-a+.00000000000000000006;
      b=z*a-b-.00000000000000000020;
      a=z*b-a+.00000000000000000066;
      b=z*a-b-.00000000000000000216;
      a=z*b-a+.00000000000000000721;
      b=z*a-b-.00000000000000002443;
      a=z*b-a+.00000000000000008441;
      b=z*a-b-.00000000000000029752;
      a=z*b-a+.00000000000000107116;
      b=z*a-b-.00000000000000394564;
      a=z*b-a+.00000000000001489474;
      b=z*a-b-.00000000000005773537;
      a=z*b-a+.00000000000023030657;
      b=z*a-b-.00000000000094784973;
      a=z*b-a+.00000000000403683207;
      b=z*a-b-.00000000001785432348;
      a=z*b-a+.00000000008235329314;
      b=z*a-b-.00000000039817923621;
      a=z*b-a+.00000000203088939238;
      b=z*a-b-.00000001101482369622;
      a=z*b-a+.00000006418902302372;
      b=z*a-b-.00000040756144386809;
      a=z*b-a+.00000287536465397527;
      b=z*a-b-.00002321251614543524;
      a=z*b-a+.00022505317277986004;
      b=z*a-b-.00287636803664026799;
      a=z*b-a+.06239591359332750793;
      double p;
      p=.5*z*a-b    +1.06552390798340693166;
      synrad=p*sqrt(M_PI_2/x)/exp(x);
    }
  }
  return synrad;
}

double syn_gen_photon_energy_normalized(LocalParticle *part)
{
  // initialize constants used in the approximate expressions
  // for SYNRAD   (integral over the modified Bessel function K5/3)
  //  xmin = 0.;
  double const xlow = 1.; 
  double const a1 = 2.149528241534391; // Synrad(1.e-38)/pow(1.e-38,-2./3.);
  double const a2 = 1.770750801624037; // Synrad(xlow)/exp(-xlow);
  double const c1 = 0.; // 
  double const ratio = 0.908250405131381;
  double appr,exact,result;
  do { 
    if(LocalParticle_generate_random_double(part) < ratio) { // use low energy approximation
      result=c1+(1.-c1)*LocalParticle_generate_random_double(part);
      double tmp = result*result;
      result*=tmp;  	// take to 3rd power;
      exact=SynRad(result);
      appr=a1/tmp;
    } else {				// use high energy approximation
      result=xlow-log(LocalParticle_generate_random_double(part));
      exact=SynRad(result);
      appr=a2*exp(-result);
    }
  } while(exact < appr*LocalParticle_generate_random_double(part));	// reject in proportion of approx
  return result; // result now exact spectrum with unity weight
}

double average_number_of_photons(double beta_gamma, double kick )
{
  return 2.5/SQRT3*ALPHA_EM*beta_gamma*fabs(kick);
}

size_t syn_gen_photons(LocalParticle *part, double kick /* rad */, double length /* m */ )
{
  if (fabs(kick) < 1e-15)
    return 0;
  
  size_t nphot = 0;
  double const mass = LocalParticle_get_mass(part); // eV
  double energy = LocalParticle_get_energy(part); // eV
  double gamma = energy / mass; // 
  double beta_gamma = sqrt(gamma*gamma-1); //
  for (double n = Radiation.Exponential(); n < average_number_of_photons(beta_gamma, kick); n += Radiation.Exponential()) {
    nphot++;
    gamma = energy / mass; // TODO: check if it's gamma, or beta*gamma
    beta_gamma = sqrt(gamma*gamma-1); // that's how it is beta gamma
    double const c1 = 1.5 * 1.973269804593025e-07; // hbar * c = 1.973269804593025e-07 eV * m
    double const energy_critical = c1 * (gamma*gamma*gamma) * fabs(kick) / length; // eV
    double const energy_loss = syn_gen_photon_energy_normalized(part) * energy_critical; // eV
    if (energy_loss >= energy) {
      energy = 0.0; // eV
      break;
    }
    energy -= energy_loss; // eV
  }

  if (energy == 0.0)
    LocalParticle_set_state(part, 0);
  else
    LocalParticle_set_energy(part, energy);

  return nphot;
}

#endif /* XTRACK_SYNRAD_SPECTRUM_H */
