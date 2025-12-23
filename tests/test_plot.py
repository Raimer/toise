from matplotlib import pyplot as plt
from matplotlib import style

package_path = "/home/lhaaso/huangtq/package/toise"

print("toise path:", package_path)

style.use(package_path + "/figures/toise.mplstyle")
import warnings
from functools import partial

import numpy as np

from toise import diffuse, factory, grb, multillh, plotting, pointsource, surface_veto

from toise import factory, grb, multillh, plotting, pointsource, surface_veto

warnings.filterwarnings("ignore")  # turn off warnings

#aeffs = factory.get("Gen2-InIce-TracksOnly")
aeffs = factory.get("Fictive-Optical")

from matplotlib.colors import LogNorm

track_aeff = aeffs["unshadowed_tracks"][0]
# find the zenith band corresponding to declination 5 degrees
zi = (
    track_aeff.get_bin_edges("true_zenith_band").searchsorted(-np.sin(np.radians(5)))
    - 1
)

fig = plt.figure(figsize=(6, 4))
ax = fig.subplots()
im = ax.pcolor(
    np.degrees(track_aeff.get_bin_edges("reco_psi")[:-1]),
    track_aeff.get_bin_edges("reco_energy"),
    pointsource.SteadyPointSource(track_aeff, livetime=1, zenith_bin=zi).expectations(
        ps_gamma=-2
    ),
    norm=LogNorm(vmin=1e-3),
)
ax.semilogy()
ax.set_ylim(1e3, 1e7)
ax.set_xlim(0, 1)
ax.set_xlabel("Opening angle $\Psi$ (degrees)")
ax.set_ylabel("Energy proxy (GeV)")
dec = np.degrees(np.arcsin(-track_aeff.get_bin_edges("true_zenith_band")))
ax.set_title("${:.1f} \leq \delta < {:.1f}$ deg".format(*dec[zi + 1 : zi - 1 : -1]))
cbar = plt.colorbar(im, ax=ax).set_label("Signal events per year")
fig.savefig("expected_counts.pdf")

muon_aeff = aeffs["unshadowed_tracks"][1]

import matplotlib.gridspec as gridspec

fig = plt.figure()
gs0 = gridspec.GridSpec(1, 2, figure=fig)
axes = [fig.add_subplot(gs0[0]), fig.add_subplot(gs0[1])]
for i, dec in enumerate([5.0, -25.0]):
    zi = (
        track_aeff.get_bin_edges("true_zenith_band").searchsorted(
            -np.sin(np.radians(dec))
        )
        - 1
    )
    dec = np.degrees(np.arcsin(-track_aeff.get_bin_edges("true_zenith_band")))
    axes[i].set_title(
        "${:.1f} \leq \delta < {:.1f}$ deg".format(*dec[zi + 1 : zi - 1 : -1])
    )
    exes = [
        (
            r"Astro $\nu$ (signal)",
            pointsource.SteadyPointSource(track_aeff, livetime=1, zenith_bin=zi)
            .expectations(ps_gamma=-2)
            .sum(axis=0),
        ),
        (
            r"Atm $\nu$ (background)",
            diffuse.AtmosphericNu.conventional(track_aeff, livetime=1)
            .point_source_background(zenith_index=zi)
            .expectations.sum(axis=0),
        ),
        (
            r"Atm $\mu$ (background)",
            surface_veto.MuonBundleBackground(muon_aeff, livetime=1)
            .point_source_background(
                zenith_index=zi, psi_bins=track_aeff.get_bin_edges("reco_psi")[:-1]
            )
            .expectations.sum(axis=0),
        ),
    ]
    for label, ex in exes:
        axes[i].plot(
            *plotting.stepped_path(
                np.degrees(track_aeff.get_bin_edges("reco_psi")[:-1]), ex
            ),
            label=label
        )
    axes[i].semilogy(nonpositive="clip")
    axes[i].set_xlabel("Opening angle $\Psi$ (degrees)")
axes[0].set_ylabel("Number of Events")
axes[1].legend()
plt.tight_layout()
fig.savefig("S_v_B_bands.pdf")

# test numerical neutrino spectrum

# neutrino energy 10**np.linspace(log(E_min/GeV), log(E_max/GeV), N_points)
# E_min >= 10 GeV, E_max <= 1e11 GeV
E_nu = 10 ** np.linspace(4, 11, 101)
# single flavor neutrino spectrum [1e-15 GeV^-1 cm^-2 s^-1]
spectrum_nu = (E_nu / 1e3) ** (-2.0)

fig = plt.figure()
gs0 = gridspec.GridSpec(1, 2, figure=fig)
axes = [fig.add_subplot(gs0[0]), fig.add_subplot(gs0[1])]
for i, dec in enumerate([5.0, -25.0]):
    zi = (
        track_aeff.get_bin_edges("true_zenith_band").searchsorted(
            -np.sin(np.radians(dec))
        )
        - 1
    )
    dec = np.degrees(np.arcsin(-track_aeff.get_bin_edges("true_zenith_band")))
    axes[i].set_title(
        "${:.1f} \leq \delta < {:.1f}$ deg".format(*dec[zi + 1 : zi - 1 : -1])
    )
    exes = [
        (
            r"Astro $\nu$ (signal)",
            pointsource.SteadyPointSource(track_aeff, livetime=1, zenith_bin=zi)
            .expectations(ps_energy=E_nu, ps_spectrum=spectrum_nu)
            .sum(axis=0),
        ),
        (
            r"Atm $\nu$ (background)",
            diffuse.AtmosphericNu.conventional(track_aeff, livetime=1)
            .point_source_background(zenith_index=zi)
            .expectations.sum(axis=0),
        ),
        (
            r"Atm $\mu$ (background)",
            surface_veto.MuonBundleBackground(muon_aeff, livetime=1)
            .point_source_background(
                zenith_index=zi, psi_bins=track_aeff.get_bin_edges("reco_psi")[:-1]
            )
            .expectations.sum(axis=0),
        ),
    ]
    for label, ex in exes:
        axes[i].plot(
            *plotting.stepped_path(
                np.degrees(track_aeff.get_bin_edges("reco_psi")[:-1]), ex
            ),
            label=label
        )
    axes[i].semilogy(nonpositive="clip")
    axes[i].set_xlabel("Opening angle $\Psi$ (degrees)")
axes[0].set_ylabel("Number of Events")
axes[1].legend()
plt.tight_layout()
fig.savefig("S_v_B_bands_discrete_spectrum_mode.pdf")
