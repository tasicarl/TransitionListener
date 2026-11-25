# Nucleation rates from the three-dimensional effective theory

The modules of this folder replace the four-dimensional rate of TransitionListener, from the bounce of the
daisy-resummed effective potential, by the rate of the three-dimensional effective theory obtained by dimensional
reduction, at leading order or at next-to-leading order. A model takes this rate by mixing in
`ThreeDimensionalRate` of `one_field_rate_3d.py`; its run configuration then has `rate_mode = "model"`. The physics
is described in the implementation note "The nucleation rate in the three-dimensional effective theory" of the
documentation, and these notes will be extended in version 3 of TransitionListener.

## What is general and what assumes a model

| file | what it does | scope |
|---|---|---|
| `one_field_rate_3d.py` | `ThreeDimensionalRate`: at each temperature asks the model for its matched 3D parameters, takes the leading-order action, adds the NLO correction and hands the result to the percolation solver as the action | any model with one scalar field that supplies `soft_parameters(T)` and the mass of its heaviest field-dependent mode |
| `abelian_higgs_3d.py` | the soft potential, the bounce, the fluctuation channels, the double-counting subtraction, the two-loop potential, the operators of dimension six and their assembly into the exponent | one scalar with an abelian gauge field (abelian Higgs); its parameters come from the model |
| `abelian_determinants.py` | functional determinants by the Gelfand–Yaglom method: single channels (with BubbleDet) and the coupled block of the Goldstone mode and the gauge field (own solver) | the single channels are general; the coupled block is that of an abelian gauge field |
| `abelian_lo_bounce_table.py` | the leading-order 3D action from one shipped table of a universal function (`tab_data/lo_bounce_tables/`), command `tl-bounce-table` to rebuild it | the potential of one scalar with degenerate abelian gauge bosons |
| `nlo_tables.py` | the NLO correction per temperature, its tables on a grid of couplings (`tab_data/nlo_grids/`), and their interpolation at run time; command `tl-nlo-table` (`interface/nlo_cli.py`) | general: everything model-specific comes through the model (`nlo_grid_name`, `nlo_grid_point`, `input_scale_GeV`, `two_loop_potential`, `two_loop_kinetic`) |
| `validity.py` | the ratios that decide whether the 3D theory applies at the percolation temperature, and the warning | general; the ratios a model can supply are listed in the module |
| `running_coupling_bounce.py` | the four-dimensional rate with the couplings run to a scale proportional to T (variant N4 of the paper): the bounce on the model's `V_rg`, its action from a shipped table (`tab_data/rg_action_tables/`, for the model with fermions) or solved directly, and the construction of the table, which stores ln(T/mu0) relative to a cheap reference action of the model | general; the potential `V_rg` and the reference action come from the model |
| `rg.py` | the couplings at every temperature from one dense solution of the renormalisation-group equations | general; the beta functions come from the model |

The model-specific parts live in the model files: the matching relations, generated with DRalgo
(`models/dr_conformal_u1_fermions_matching.py`, `models/dr_conformal_u1_no_fermions_matching.py`), and the models
that use them (`models/TL_conformal_u1_fermions_3d.py`, `models/TL_conformal_u1_no_fermions_3d.py`). These are the
only models with this rate so far, and the only ones with shipped NLO tables.

The compiled kernels of the bounce table and of the coupled determinant are in `../compiled/` (`bounce_table.py`,
`mixing.py`).
