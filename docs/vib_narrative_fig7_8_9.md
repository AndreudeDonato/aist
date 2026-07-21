# Vibrational thermodynamics narrative (Figs. 7 → 8 → 9)

Notes for writing the paper section on vibrational / ZPE effects in the PBE grand-canonical ensemble. Figure order already matches the story:

1. **Fig. 7** — vib as an *energy* preference  
2. **Fig. 8** — vib reshuffles *populations* among near isomers  
3. **Fig. 9** — does that preference change the *observable* oxidation on the phase diagram?

Micro → meso → macro. State early that Fig. 9 is a **filtered consequence** of Figs. 7–8, not a direct colour-map of the stoichiometry curves in Fig. 7.

**Scripts / outputs**

| Figure | Script | Output |
|---|---|---|
| 7 — Energy preference | `figures/fig07_vib_zpe_stoichiometry.py` | `fig07_vib_zpe_stoichiometry.svg` |
| 8 — Populations | `figures/fig08_min_pop_at_low_po2.py` | `fig08_min_pop_at_low_po2.svg` |
| 9 — Observable Δ maps | `figures/fig09_oxidation_delta_maps.py` | `fig09_oxidation_delta_maps.svg` |

Full Ir₃…Ir₉ version of the Δ maps: `supp_ir_size_oxidation_delta_maps.svg`.  
Ambient line cuts of oxidation Δ: `supp_oxidation_delta_by_ir.svg`.

---

## Fig. 7 — Magnitude on the energy scale

**What it shows**

- Top: Boltzmann \(\langle\Delta\mathrm{ZPE}\rangle\) vs vibrational DOF \(N_{\mathrm{vib}}\) (by Ir size).  
- Below: Boltzmann reaction \(\langle\Delta G_{\mathrm{vib}}\rangle(T)\) for Ir₃, Ir₆, Ir₉, coloured by \(n_O/n_{\mathrm{Ir}}\).

**How \(\Delta G_{\mathrm{vib}}\) is defined**

Formation free-energy contribution **per Ir** (balanced reaction Ir + O₂ → cluster), including cluster vib/ZPE and the O₂ reservoir vib/ZPE terms. Lower \(\langle\Delta G_{\mathrm{vib}}\rangle\) = vib thermodynamically favours that stoichiometry more.

**Key messages**

- Curves are essentially **intensive** (similar across Ir₃ / Ir₆ / Ir₉) because the reaction is normalised per Ir.  
- **Pressure barely matters** for these curves: harmonic vib/ZPE depend on \(T\), not \(p\). Pressure only enters weakly via isomer weights inside a stoichiometry bucket.  
- **Low \(T\)**: vib (mainly ZPE) slightly favours **more reduced** stoichiometries; spreads are small (few–tens of meV).  
- **From ~300 K onward**: ordering crosses over toward **more oxidized** stoichiometries; gaps grow with \(T\) as \(-T S_{\mathrm{vib}}\) from soft oxide modes takes over. ~300 K is an **onset / crossover**, not a hard switch (size- and stoich-dependent).

**Paper angle**

Establish that vibrations are a real but modest energy correction with a clear \(T\)-dependent stoichiometry preference — *before* asking whether the GC ensemble oxidation moves.

---

## Fig. 8 — Vib × ensemble: populations

**What it shows**

For Ir₃, Ir₆, Ir₉ at ambient \(p\mathrm{O_2}\), ~273 K and ~500 K: population (%) of the five lowest-energy structures (order fixed by **novib** \(G\)). Blue = no vib/ZPE; orange = full vib+ZPE.

**Key messages**

- Same electronic landscape, but vib/ZPE **reweights** which near isomers carry the population.  
- This is the bridge between Fig. 7 (who vib likes on an energy scale) and Fig. 9 (whether average oxidation changes): vib often moves weight among **near-degenerate** structures without a huge stoichiometry jump.  
- Fixed \((T,p)\) snapshots — local competition among low-\(G\) isomers, not the full phase diagram.

**Paper angle**

“Even when the global-minimum stoichiometry is unchanged, vibrational terms redistribute probability among the low-lying manifold.” Links naturally to why ensemble averaging matters more than vib alone in Fig. 9’s left column.

---

## Fig. 9 — Observable: phase-diagram oxidation differences

**What it shows** (PiYG divergent \(\Delta(n_O/n_{\mathrm{Ir}})\), Ir₃ | Ir₆ | Ir₉)

| Column | Definition |
|---|---|
| Left | Full (vib + ZPE + **ensemble**) − no vib/ZPE, **single minimum** |
| Right | Full (vib + ZPE + **ensemble**) − no vib/ZPE, **ensemble** |

Positive Δ = full thermo more oxidized than the reference.

### Right column — fair vib comparison

Ensemble full vs ensemble novib. Small shifts of phase boundaries / mean oxidation. Typical message:

- Vib/ZPE is a **weak correction** on the ensemble oxidation ladder (tenths in \(n_O/n_{\mathrm{Ir}}\), not order-one jumps).  
- At **low \(T\)**: excluding vib tends to **slight overoxidation** → with vib the ensemble sits a bit **more reduced**. This **agrees** with Fig. 7’s low-\(T\) / ZPE preference.  
- At **high \(T\)**, especially **low \(p\mathrm{O_2}\)**: maps need **not** show the opposite of Fig. 7 (vib favouring oxidized). That is **not** a contradiction — see below.

### Left column — do not read as “vib only”

Full ensemble minus novib **single minimum** mixes **two** changes:

1. vib + ZPE on vs off  
2. Boltzmann average vs one winner  

Large structure here is mostly **ensemble vs minimum**, with vib entangled. It should **not** be expected to correlate with Fig. 7’s stoichiometry curves. Use it to stress that dropping ensemble character changes the picture far more than turning vib on/off within the ensemble.

---

## Resolving Fig. 7 vs Fig. 9 at high \(T\)

| Figure | Question |
|---|---|
| Fig. 7 | Among *all* stoichiometries, who does vib prefer? |
| Fig. 9 (right) | Among stoichiometries that are *actually competitive* in the GC ensemble, does turning vib off raise or lower \(\langle n_O/n_{\mathrm{Ir}}\rangle\)? |

At **high \(T\), low \(p\mathrm{O_2}\)** those sets differ:

1. **Oxidized phases are off-stage.** \(\mu_{\mathrm{O_2}}(T,p)\) already forces the system deep into the reduced ladder. Fig. 7’s vib bonus for high \(n_O/n_{\mathrm{Ir}}\) cannot pull oxides back against a large electronic / chemical-potential penalty.  
2. **Δ only sees the local fight** among near-degenerate **reduced** phases / boundary shifts — not “oxides would like vib more globally.”  
3. **O₂ gas entropy (trans/rot in \(\mu\))** dominates reduction with \(T\) at low \(p\). Cluster \(\Delta G_{\mathrm{vib}}\) is a small correction on top.  
4. Large phase jumps make Δ noisy: spikes reflect **boundary shifts**, not a smooth readout of Fig. 7.

So:

- **Low \(T\)**: Fig. 7 (vib → reduced) and Fig. 9 right (no vib → slight overoxidation) agree.  
- **High \(T\)**: Fig. 7’s “vib → oxidized” remains true on the energy scale but is often **irrelevant** to ensemble Δ at low \(p\), because those oxidized stoichiometries are not competing.

Where the high-\(T\) Fig. 7 branch *can* show up in oxidation: milder landscapes (e.g. ambient \(p\mathrm{O_2}\) line cuts in `supp_oxidation_delta_by_ir`), where oxidized phases stay nearer degeneracy.

---

## One-paragraph skeleton for the paper

Vibrational free energies (Fig. 7) are intensive and favour slightly more reduced stoichiometries at low temperature (ZPE), crossing over toward more oxidized species above roughly room temperature as vibrational entropy grows; the energy scales involved are modest. These preferences redistribute Boltzmann weight among near-lying isomers (Fig. 8). On the grand-canonical phase diagram (Fig. 9), the corresponding change in ensemble \(\langle n_O/n_{\mathrm{Ir}}\rangle\) is therefore a filtered observable: within the ensemble, vib/ZPE only slightly shifts boundaries (typically toward slightly less oxidation at low \(T\) when vib is included), whereas comparing a single global minimum to the full ensemble produces much larger differences. At high \(T\) and low \(p\mathrm{O_2}\), the microscopic preference for oxidized clusters in Fig. 7 does not map onto the oxidation Δ maps, because those stoichiometries are no longer competitive under the oxygen chemical potential.

---

## Sign cheat sheet

**Fig. 7**  
Lower \(\langle\Delta G_{\mathrm{vib}}\rangle\) ⇒ vib stabilises that stoichiometry more.

**Fig. 9 (Δ maps)**  
\(\Delta =\) full − novib.  
- \(\Delta > 0\): full more oxidized than novib ⇒ vib increases oxidation.  
- \(\Delta < 0\): full more reduced than novib ⇒ vib decreases oxidation (matches low-\(T\) “novib overoxidizes”).

**Line-cut convention** (`supp_oxidation_delta_by_ir`, variant − VZE)  
Positive \\\\E − VZE ⇒ novib more oxidized than full ⇒ same low-\(T\) story as \(\Delta < 0\) on the Fig. 9 maps.

---

## What not to claim

- Do not claim Fig. 9’s left column isolates vib.  
- Do not claim Fig. 7’s high-\(T\) oxidized preference must appear as “full more oxidized” everywhere on Fig. 9.  
- Do not lean on pressure as an explanation for Fig. 7 itself.  
- Do not oversell vib relative to ensemble averaging: the striking shifts are usually **without ensemble**, not vib vs novib inside the ensemble.
