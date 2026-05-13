"""This module contains routines to compute the possible transitions
between phases.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import optimize

from . import pathDeformation
from . import errors

from . import console, print

from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns as RichColumns
from rich.box import ROUNDED

from transitionlistener.nucleation import computeNucleationTemperature
from transitionlistener.generic_potential import generic_potential

if TYPE_CHECKING:
    from transitionlistener.phases import PhaseInfo


@dataclass
class TransitionInfo():
    """Object containing the information about one possible phase transition."""
    Tcrit: float
    """Critical temperature where the two minima are degenerate"""
    high_vev: np.ndarray
    """Field values in the high phase"""
    high_phase: int
    """Key of the high phase"""
    low_vev: np.ndarray
    """Field values in the low phase"""
    low_phase: int
    """Key of the low phase"""
    type: int
    """Type of transition: 1 for first order, 2 fo second order (continuous)"""
    Tnuc: float = 0.0
    """Nucleation temperature."""
    Tperc: float = 0.0
    """Percolation temperature."""
    full_tunneling_info = {}
    """Save the temperatures and corresponding actions from the nucleation
    computation."""
    action_Tnuc: float = 0.0
    """Action at the nucleation."""
    instanton = {}
    """Instanton solution at the nucleation temperature."""
    step: int = 0
    """Ordering number of the possible transitions"""
    total_steps: int = 0
    """Total number of transitions."""
    derived_params = {}
    """Computed observables."""


def _transition_sort_temperature(transition: TransitionInfo) -> tuple[float, float]:
    """Sort transitions by descending nucleation temperature, falling back to ``Tcrit``.

    Seedless percolation can keep first-order transition candidates with
    ``Tnuc=None``. Those transitions still need a stable ordering, so we fall
    back to ``Tcrit`` when no finite ``Tnuc`` is available.
    """
    tnuc = transition.Tnuc
    tcrit = transition.Tcrit
    tnuc_key = float(tnuc) if tnuc is not None and np.isfinite(tnuc) else float("-inf")
    tcrit_key = float(tcrit) if tcrit is not None and np.isfinite(tcrit) else float("-inf")
    return tnuc_key, tcrit_key


def _format_temperature_gev(temperature: float | None, conversion_factor: float) -> str:
    """Return a readable GeV string for finite temperatures and ``None`` otherwise."""
    if temperature is None or not np.isfinite(temperature):
        return "None"
    return str(temperature * conversion_factor)


class Transitions():
    """Compute the transition path from the high-T phase to the 0-T phase
    for a given set of phases.

    Class containing the routines to find the transitions between
    all possible pairs of phases.

    Parameters
    ----------
    phases : list["PhaseInfo"]
        List of the phases as PhaseInfo objects.
    pot : generic_potential
        The potential.
    verbose : bool, optional
        Set the outputlevel.
    """

    def __init__(self, phases: list["PhaseInfo"], pot: generic_potential,
                 verbose: bool = False):
        """Trace the transition network and determine nucleation temperatures.

        Parameters
        ----------
        phases : list[PhaseInfo]
            List of the phases as PhaseInfo objects.
        pot : generic_potential
            The potential.
        verbose : bool, optional
            Set the outputlevel.
        """
        self.phases = phases
        self.pot = pot
        self.V = pot.Vtot
        self.dV = pot.gradV
        self.d2V = pot.d2V
        self.verbose = verbose

        if self.phases is None or len(self.phases) == 0:
            msg = "Cannot find phase transitions, because no phases were found."
            raise errors.NoPhases(msg)
        if len(self.phases) == 1:
            msg = "Cannot find phase transitions, because only one phase was found."
            raise errors.OnlyOnePhase(msg)

        # assemble the tunneling parameters:
        self.tunnelFromPhase_args = {"Ttol": pot.config.tracingConf.nucleation_Ttol,
                                     "approximate_strength_threshold": pot.config.tracingConf.approx_strength_threshold,
                                     "nuclCriterion": pot.nucleationCriterion,
                                     "fullTunneling_params": pot.config.tracingConf.tunneling_params}

        self.conversionFactor = pot.conversionFactor
        
        if verbose:
            console.print("\n[bold green]Finding phase transitions...[/bold green]")

        transitionsTnuc = self.findTransitionPathFromTnuc(verbose=verbose)
        self.transitions = transitionsTnuc
        
        if verbose:
            msg = (
                f"\n[bold green]Found {len(transitionsTnuc)} possible "
                f"phase transition(s) ...[/bold green]"
            )
            console.print(msg)
            
            self.printTransitions(transitionsTnuc)
        
        if len(transitionsTnuc) == 0:
            raise errors.NoTransitionFound("No transition found.")
        
        if all(tr.type == 2 for tr in transitionsTnuc):
            raise errors.OnlySecondOrderTransitionsError(
                "Only second-order transition(s) found. No first-order transitions.")


    def __str__(self):
        """Pretty-print the discovered transitions and return an empty string."""
        self.printTransitions(self.transitions)
        return ""
        
    def __iter__(self):
        """Enable iteration over the list of transitions."""
        return iter(self.transitions)

    def __getitem__(self, key: str):
        """Allow dictionary-style access via the transition index."""
        return self.transitions[key]

    def __len__(self):
        """Return the number of transitions that were found."""
        return len(self.transitions)

    def keys(self):
        """Expose the valid indices for the transition list."""
        return self.transitions.keys()

    def findAllTransitions(self) -> dict[TransitionInfo]:
        """Find all the possible transitions between coexisting phases
        by computing the nucleation temperatures.

        Parameters
        ----------
        phases : list[PhaseInfo]
        V : func
        dV : func

        Returns
        -------
        dict[TransitionInfo]
            List with TransitionInfo objects.
        """
        pot = self.pot
        phases = self.phases
        verbose = self.verbose
        tunnelFromPhase_args = self.tunnelFromPhase_args
        seedless = pot.config.percolationConf.algorithm_mode == "adaptive_step_size"

        # This below could miss transitions in 2D cases where there is no critical
        # temperature but the phases still coexist!
        transitions = self.findTransitionPathFromTnuc(verbose=verbose)

        if transitions is None:
            raise errors.NoTransitionFound("No transition found.")

        if verbose:
            message = f"[bold green]Found {len(transitions)} possible transition(s):[/bold green]"
            for tr in transitions:
                message += f"\n{tr.high_phase} -> {tr.low_phase} at Tcrit = {tr.Tcrit * self.conversionFactor} GeV"
            console.print()
            console.print(Panel.fit(message,border_style="green"))
        
        transitions_dict = []
        TooMuchSupercooling = False
        for tr in transitions:
            if tr.type != 1:
                # This is a second order transition. Still add it to the list
                # of transitions, but don't try to find the tunneling solution.
                second_order_dict = self.makeSecondOrderDict(tr)
                transitions_dict.append(second_order_dict)
                continue
            Tmax = tr.Tcrit
            if verbose:
                print("\n"+"#"*50)
                print(f"Investigating transition with Tcrit = {tr.Tcrit * self.conversionFactor} GeV" + \
                      f" from phase {tr.high_phase:} to phase {tr.low_phase:}")
            high_phase = phases[tr.high_phase]
            low_phase = phases[tr.low_phase]
            try:
                tunn = computeNucleationTemperature(self.V, self.dV, self.d2V, high_phase, low_phase,
                                                    self.conversionFactor, verbose=verbose,
                                                    allow_second_order_fallback=not seedless,
                                                    **tunnelFromPhase_args)
                # sometimes a second order transition slips through the phase tracing
                if tunn['trantype'] == 2:
                    tr.type = 2
                    tr.action = np.nan
                    transitions_dict.append(tr)
                else:
                    tr.full_tunneling_info = tunn["full_tunneling_info"]
                    tr.Tnuc = tunn["Tnuc"]
                    tr.low_phase = tunn["low_phase"]
                    tr.high_phase = tunn["high_phase"]
                    tr.instanton = tunn["instanton"]
                    tr.type = tunn["trantype"]
                    tr.action = tunn["action"]
                    transitions_dict.append(tr)
            except errors.NucleationError as err:
                if verbose:
                    print("Nucleation error: ", err)
                if seedless:
                    tr.Tnuc = None
                    tr.action = np.nan
                    tr.full_tunneling_info = {}
                    tr.instanton = {}
                    transitions_dict.append(tr)
                else:
                    TooMuchSupercooling = True
                    continue
            except Exception as err:
                raise errors.NucleationError("An unexpected error occurred " + 
                                             "while trying to find the tunneling solution: "
                                             + str(err))

        # Check if we have found any transitions
        if len(transitions_dict) == 0:
            if TooMuchSupercooling:
                raise errors.TooMuchSupercoolingError()
            raise errors.NoTransitionFound("No transition found.")
        
        # Check if only second-order transitions were found
        if all(tr.type == 2 for tr in transitions_dict):
            raise errors.OnlySecondOrderTransitionsError(
                "Only second-order transition(s) found. No first-order transitions.")

        # Check if there are transitions in the list, that are not possible
        # because they already happened through a (series of) previous transition(s).
        # For instance, if nucleation temperatures for the transitions 0 -> 1, 1 -> 2,
        # and 0 -> 2 are found, but 0 -> 1 has already happened when 0 -> 2 would nucleate,
        # 0 -> 2 cannot happen and we can remove the 0 -> 2 transition from the list.
        # This is done by going through the list of transitions, starting from
        # the highest Tnuc and checking if a given transition is would start
        # from a phase, that has already disappeared:
        disappeared_phases = set()
        # order the transitions by Tnuc: the highest Tnuc first
        transitions_dict = sorted(transitions_dict, key=_transition_sort_temperature, reverse=True)

        # Have to check if the first transition is second order, if yes ->
        # replace highT_phase_key with the one it second order transitions to!
        highT_phase_key = [key for key in phases.keys() if phases[key].T[-1] == pot.Tmax][0]
        for tr in transitions_dict:
            if tr.high_phase == highT_phase_key and tr.type == 2:
                highT_phase_key = tr.low_phase

        for i, tr in enumerate(transitions_dict):
            # Check if the first FOPT in the dictionary actually starts from
            # the high-T phase of the model. If not, we throw an error.

            # Note: this is only a hotfix for a specific physics bug in the vev flip flop model.
            # There are cases in which this check here leads to the wrong results. What should be
            # done instead is to only check one transition at a time, starting from the high-T phase.
            # This will need some restructuring of the code, so for now we just
            # check the first transition in the dictionary.
            if pot.config.tracingConf.do_high_T_phase_check:
                if i == 0:
                    if tr.high_phase != highT_phase_key and tr.type == 1:
                        raise errors.WrongHighTPhaseError(
                            "The first transition in the transition dictionary " + 
                            "does not start from the high-T phase of the model. " +
                            "This is unexpected and indicates a bug in the code.")

            # If the starting (high) phase of the transition is in the set of
            # disappeared phases, we can remove this transition from the list
            if tr.high_phase in disappeared_phases:
                if verbose:
                    print(
                        "\n WARNING: Removing the transition "
                        f"{tr.high_phase} -> {tr.low_phase} at Tnuc = "
                        f"{_format_temperature_gev(tr.Tnuc, self.conversionFactor)} GeV from the dictionary "
                        "because the high phase has already disappeared in a "
                        "previous transition."
                    )
                # transitions_dict[i] = None
                transitions_dict.remove(transitions_dict[i])
            else:
                # If the high phase is not in the set of disappeared phases, we can add it to the set
                disappeared_phases.add(tr.high_phase)

        # Enumerate the transitions and save the order in the dictionary
        for i, tr in enumerate(transitions_dict):
            tr.step = i + 1
            tr.total_steps = len(transitions_dict)
        
        # Output the transitions
        if verbose:
            print("\nFound %d transition(s): \n" % len(transitions_dict))
            self.printTransitions(transitions_dict)
            
        return transitions_dict

    def printTransitions(self, transitions: dict[TransitionInfo]) -> None:
        """Pretty-printing of the transitions

        Parameters
        ----------
        transitions : dict[TransitionInfo]
            List with TransitionInfo objects.
            """
        tables = []
        for tr in transitions:
            if tr.type == 1:
                table = Table(title="First-order phase transition", title_justify="left", box=ROUNDED)
                table.add_column("Parameter", style="cyan", no_wrap=True)
                table.add_column("Value", style="orange1")
                table.add_row("Tnuc / GeV", _format_temperature_gev(tr.Tnuc, self.pot.conversionFactor))
                table.add_row("high phase", str(tr.high_phase))
                table.add_row("low phase", str(tr.low_phase))
                table.add_row("high vev / GeV", str(tr.high_vev * self.pot.conversionFactor))
                table.add_row("low vev / GeV", str(tr.low_vev * self.pot.conversionFactor))
                table.add_row("action / GeV", str(tr.action_Tnuc * self.pot.conversionFactor))
                tables.append(table)
            elif tr.type == 2:
                # This is a second-order transition
                table = Table(title="Second-order phase transition", title_justify="left", box=ROUNDED)
                table.add_column("Parameter", style="cyan", no_wrap=True)
                table.add_column("Value", style="orange1")
                table.add_row("Tcrit / GeV", _format_temperature_gev(tr.Tcrit, self.pot.conversionFactor))
                table.add_row("high phase", str(tr.high_phase))
                table.add_row("low phase", str(tr.low_phase))
                table.add_row("high vev / GeV", str(tr.high_vev * self.pot.conversionFactor))
                table.add_row("low vev / GeV", str(tr.low_vev * self.pot.conversionFactor))
                tables.append(table)
        console.print(RichColumns(tables))

    def secondOrderTrans(self, V: callable, high_phase: "PhaseInfo",
                         low_phase: "PhaseInfo", verbose=False,
                         degeneracyTest=True) -> TransitionInfo:
        """Create the TransitionInfo object for a second order transition.

        Parameters
        ----------
        V : callable
            The effective potential function.
        high_phase : PhaseInfo
            The phase information for the high-temperature phase
        low_phase : PhaseInfo
            The phase information for the low-temperature phase
        verbose : bool
            Set the output level.
        degeneracyTest : bool, optional
            If true, search for Tcrit, where the to minima are degenerate.
            Note, that this causes errors for hidden second order transitions

        Returns
        -------
        TransitionInfo:
            The transition information.
            
        Notes
        -----
        A second order transition is identified by finding the critical temperature
        Tcrit where the two phases are degenerate in energy. At this temperature, the potential
        values of the two phases are equal. If such a temperature is found and the 
        degeneracy check passes, a TransitionInfo object is created to represent the
        second order transition.
        """
        # Calculate the critical temperature Tcrit, where the two phases are
        # degenerate. This is done by finding the temperature where the
        # potential is equal for the two phases. This is done by using
        # optimize.brentq to find the root of the function
        def f(T): return V(high_phase.valAt(T), T) - V(low_phase.valAt(T), T)

        Tcrit = np.nan
        
        try:
            Tcrit = optimize.brentq(f, high_phase.T[0], low_phase.T[-1], disp=False)
            if verbose:
                print("Second order transition between phases %s and %s at Tcrit = %g GeV" \
                      % (high_phase.key, low_phase.key, Tcrit * self.conversionFactor))
                print("high_phase.valAt(Tcrit) = ", high_phase.valAt(Tcrit) * self.conversionFactor, "GeV")
                print("low_phase.valAt(Tcrit) = ", low_phase.valAt(Tcrit) * self.conversionFactor, "GeV")
                print("V(high_phase.valAt(Tcrit), Tcrit) = ", V(high_phase.valAt(Tcrit), Tcrit) \
                      * self.conversionFactor**4, "GeV^4")
                print("V(low_phase.valAt(Tcrit), Tcrit) = ", V(low_phase.valAt(Tcrit), Tcrit) \
                      * self.conversionFactor**4, "GeV^4\n")
        except ValueError as err:
            if verbose:
                lower = high_phase.T[0] * self.conversionFactor
                upper = low_phase.T[-1] * self.conversionFactor
                print(
                    "The critical temperature could not be found in the range "
                    f"({lower}, {upper}) GeV between the phases "
                    f"{high_phase.key} and {low_phase.key}. This is likely a "
                    "problem due to the numerical precision used in phase tracing."
                )
                print("ValueError: ", err)
            Tcrit = np.nan

        # no Tcrit found:
        if np.isnan(Tcrit):
            Tcrit = low_phase.Tmax
            transition = TransitionInfo(
                Tcrit=Tcrit,
                high_vev=high_phase.X[0],
                high_phase=high_phase.key,
                low_vev=low_phase.X[0],
                low_phase=low_phase.key,
                type=2,
                Tnuc=Tcrit,
                Tperc=Tcrit)
        elif np.abs(V(high_phase.valAt(Tcrit), Tcrit) / V(low_phase.valAt(Tcrit), Tcrit) - 1) < 1e-4:
            # Check if the two phases are actually degenerate at Tcrit
            # This is a second order transition
            if verbose:
                print("The check for degeneracy at Tcrit = ", Tcrit * self.conversionFactor,
                      "GeV succeeded. Create crossover transition object now.")
            transition = TransitionInfo(
                Tcrit=Tcrit,
                high_vev=high_phase.X[0],
                high_phase=high_phase.key,
                low_vev=low_phase.X[0],
                low_phase=low_phase.key,
                type=2,
                Tnuc=Tcrit, # this part is technically wrong, but we need a value to sort the transitions
                Tperc=Tcrit)
        else:
            if verbose:
                print("Degeneracy check failed at Tcrit = ", Tcrit * self.conversionFactor, "GeV")
            Tcrit = np.nan
            transition = TransitionInfo(
                Tcrit=np.nan,
                high_vev=high_phase.X[0],
                high_phase=high_phase.key,
                low_vev=low_phase.X[0],
                low_phase=low_phase.key,
                type=2,
                Tnuc=np.nan,
                Tperc=np.nan)
        return transition

    def makeSecondOrderDict(self, transition: TransitionInfo) -> dict:
        """
        Create a dictionary describing a second-order phase transition.
        """
        transition_dict = TransitionInfo(
                Tcrit=transition.Tcrit,
                high_vev=transition.high_vev,
                high_phase=transition.high_phase,
                low_vev=transition.low_vev,
                low_phase=transition.low_phase,
                type=2,
                Tnuc=np.nan,
                Tperc=np.nan)
        return transition_dict

    def sortForSmallestAction(self, V, phases, startPhase, childPhases) -> list:
        """Estimate the action by computing the action for the linearised
        potential between the start phase and the child phases.
        Return a sorted list of the child phases after the
        smallest action approximation

        Parameters
        ----------

        Returns
        -------

        .. todo:: implement

        """
        pass

    def getUniqueTunnelingPhases(self, V : callable, phases : list["PhaseInfo"],
                                 startPhase : "PhaseInfo",
                                 childPhases : list["PhaseInfo"]) -> list["PhaseInfo"]:
        """Check if there are equivalent tunneling paths to different phases.

        This functions checks there are two phases which have the same distance
        to the start phase and the same potential values. If so the
        tunneling paths are considered equivalent and only one phase is returned.

        Parameters
        ----------
        V : callable
            The effective potential V(X, T)
        phases : list[PhaseInfo]
        childPhases : list
            List of phase keys

        Returns
        -------
        list :
             List with phase keys.
        """
        xtol = 1e-3
        Vtol = 1e-3
        uniqueTunnelingPhases = []
        while childPhases != []:
            p1 = childPhases.pop()
            DX1min = np.sqrt(np.sum(phases[p1].X[0] - phases[startPhase].X[0])**2)
            DV1min = np.abs(V(phases[p1].X[0], phases[p1].T[0]) -
                            V(phases[startPhase].X[0], phases[startPhase].T[0]))
            DX1max = np.sqrt(np.sum(phases[p1].X[-1] - phases[startPhase].X[-1])**2)
            DV1max = np.abs(V(phases[p1].X[-1], phases[p1].T[-1]) -
                            V(phases[startPhase].X[-1], phases[startPhase].T[-1]))
            equiv = False
            for p2 in childPhases:
                DX2min = np.sqrt(np.sum(phases[p2].X[0] - phases[startPhase].X[0])**2)
                DX2max = np.sqrt(np.sum(phases[p2].X[-1] - phases[startPhase].X[-1])**2)
                DV2min = np.abs(V(phases[p2].X[0], phases[p2].T[0]) -
                                V(phases[startPhase].X[0], phases[startPhase].T[0]))
                DV2max = np.abs(V(phases[p2].X[-1], phases[p2].T[-1]) -
                                V(phases[startPhase].X[-1], phases[startPhase].T[-1]))
                if ((abs(DV1min - DV2min) < Vtol and abs(DX1min - DX2min) < xtol) and
                    (abs(DV1max - DV2max) < Vtol and abs(DX1max - DX2max) < xtol)):
                    equiv = True
                    print("WARNING: Encountered 2 equivalent child phases. " +
                          "This implies domain walls!")
                    break
            if not equiv:
                uniqueTunnelingPhases.append(p1)

        return uniqueTunnelingPhases

    def findTransitionPathFromTnuc(self, verbose=False) -> list[TransitionInfo]:
        """Find the transition path starting from the high-T phase.

        Calculate the nucleation temperature for the transitions
        starting from the high-T phase. The function constructs
        a list of transitions with the highest possible nucleation
        temperatures.

        Parameters
        ----------
        verbose : bool
            Set the output level.

        Returns
        -------
        list[TranstionInfo]:
            A list of of transitions sorted in decreasing temperature.
        """
        transitions = []
        tunnelFromPhase_args = self.tunnelFromPhase_args
        phases = self.phases
        V = self.V
        startPhase = phases[phases.rootPhase]
        seedless = self.pot.config.percolationConf.algorithm_mode == "adaptive_step_size"

        while True:
            # check if we are in the lowest minimum
            if startPhase.childPhases == []:
                if startPhase.key == self.phases.T0Phase:
                    break
                else:
                    raise errors.WrongT0MinimumError

            childTransitions = []
            # exclude equivalent tunnelings to mirror phases
            childPhases = self.getUniqueTunnelingPhases(V, phases, startPhase.key, startPhase.childPhases)
            for i in childPhases:
                endPhase = phases[i]

                # Check for overlap between phases
                tmax = min(startPhase.Tmax, endPhase.Tmax)
                tmin = max(startPhase.Tmin, endPhase.Tmin) 

                if tmin >= tmax:
                    if endPhase.key in startPhase.low_trans:
                        childTransitions.append(
                            self.secondOrderTrans(V, startPhase, endPhase, verbose))
                        startPhase = endPhase
                        continue  # There could be other second order transitions (at slightly high temps)
                    else:
                        pass
                    continue

                if seedless:
                    transition = self.seedlessFirstOrderCandidate(
                        V,
                        startPhase,
                        endPhase,
                        tmin,
                        tmax,
                        verbose,
                    )
                    if transition is not None:
                        childTransitions.append(transition)
                    continue

                try:
                    if verbose:
                        print(f"\nTry tunneling from phase {startPhase.key} to phase {endPhase.key}")
                    tunn = computeNucleationTemperature(self.V, self.dV, self.d2V,
                                                        startPhase,
                                                        endPhase, self.conversionFactor,
                                                        verbose=verbose,
                                                        allow_second_order_fallback=not seedless,
                                                        **tunnelFromPhase_args)

                    # create TunnelingInfo object
                    tr = TransitionInfo(Tcrit=tunn['Tcrit'], Tnuc=tunn['Tnuc'],
                                        low_phase=tunn['low_phase'],
                                        high_phase=tunn['high_phase'],
                                        low_vev=tunn['low_vev'],
                                        high_vev=tunn['high_vev'],
                                        action_Tnuc=tunn['action'],
                                        type=tunn['trantype'])
                    if tr.type != 2:  # Sometimes a second order transitions slips through here
                        tr.instanton = tunn['instanton']
                    childTransitions.append(tr)
                except errors.NucleationError as err:
                    # Too much supercooling, cannot tunnel.
                    if verbose:
                        print("Nucleation error: ", err)
                    if seedless:
                        transition = TransitionInfo(
                            Tcrit=tmax,
                            Tnuc=None,
                            low_phase=endPhase.key,
                            high_phase=startPhase.key,
                            low_vev=endPhase.valAt(tmax),
                            high_vev=startPhase.valAt(tmax),
                            action_Tnuc=np.nan,
                            type=1,
                        )
                        transition.full_tunneling_info = {}
                        transition.instanton = {}
                        childTransitions.append(transition)
                    else:
                        continue
                except Exception as err:
                    raise errors.NucleationError("Unexpected nucleation error:\n" + 
                                                 str(err))

            if childTransitions == []:
                raise errors.TooMuchSupercoolingError

            childTransitions = sorted(childTransitions, key=_transition_sort_temperature, reverse=True)
            # new start phase:
            startPhase = phases[childTransitions[0].low_phase]
            transitions.append(childTransitions[0])

        for i, tr in enumerate(transitions):
            tr.step = i + 1
            tr.total_steps = len(transitions)

        return transitions

    def seedlessFirstOrderCandidate(
        self,
        V: callable,
        high_phase: "PhaseInfo",
        low_phase: "PhaseInfo",
        tmin: float,
        tmax: float,
        verbose: bool = False,
    ) -> TransitionInfo | None:
        """Build a first-order transition candidate without computing Tnuc.

        Seedless adaptive step size should not spend CPU finding a nucleation-temperature seed
        before percolation.  It only needs a plausible high/low phase pair and
        a finite overlap interval; the dynamic support-bank scouts then decide
        where action evaluations are physically relevant.
        """

        def DV(T):
            return V(high_phase.valAt(T), T) - V(low_phase.valAt(T), T)

        try:
            dv_min = DV(tmin)
            dv_max = DV(tmax)
        except Exception as err:
            if verbose:
                print(
                    "Skipping seedless candidate because the phase-overlap "
                    "potential difference could not be evaluated:",
                    err,
                )
            return None

        if dv_min < 0:
            if verbose:
                print(
                    "Skipping seedless candidate because the starting phase is "
                    "already lower than the target phase at Tmin."
                )
            return None

        candidate_temperature = tmax
        if dv_max <= 0:
            try:
                candidate_temperature = optimize.brentq(DV, tmin, tmax, disp=False)
            except ValueError:
                candidate_temperature = tmax
        elif high_phase.Tmax <= low_phase.Tmax:
            if verbose:
                print(
                    "Skipping seedless candidate because the starting phase is "
                    "higher at Tmax and also exists to at least as high a "
                    "temperature as the target phase."
                )
            return None

        if verbose:
            print(
                "Adding seedless first-order candidate from phase",
                high_phase.key,
                "to phase",
                low_phase.key,
                "with reference T =",
                candidate_temperature * self.conversionFactor,
                "GeV",
            )

        transition = TransitionInfo(
            Tcrit=candidate_temperature,
            Tnuc=None,
            low_phase=low_phase.key,
            high_phase=high_phase.key,
            low_vev=low_phase.valAt(candidate_temperature),
            high_vev=high_phase.valAt(candidate_temperature),
            action_Tnuc=np.nan,
            type=1,
        )
        transition.full_tunneling_info = {}
        transition.instanton = {}
        return transition

    def findPossibleTransitionsFromPhases(self, verbose=False) -> list[TransitionInfo]:
        """Find all possible transitions between coexisting phases.

        Parameters
        ----------
        verbose : bool
            Set the output level.

        Returns
        -------
        list[TranstionInfo]:
            A list of of transitions sorted in decreasing temperature.
        """

        phases = self.phases
        V = self.V
        start_high = False
        verbose = self.verbose

        if verbose: 
            print("\nFind all possible transitions based on the existence of a critical temperature:")
        transitions = []
        for i in phases.keys():
            for j in phases.keys():
                if i == j:
                    continue
                # Try going from i to j (phase1 -> phase2)
                phase1, phase2 = phases[i], phases[j]
                tmax = min(phase1.Tmax, phase2.Tmax)
                tmin = max(phase1.Tmin, phase2.Tmin)

                if verbose:
                    print("\nChecking transition from phase", i, "to phase", j,
                          "which coexists in the range [", tmin * self.conversionFactor,
                          ",", tmax * self.conversionFactor, "] GeV")

                if tmin >= tmax:
                    # There is no overlap between the two phases, so no transition can occur. Still, if previously
                    # the low_trans dictionary was filled, indicating the possibility of a second-order transition,
                    # we can still check for that.
                    if verbose:
                        print("There's no overlap between the phases. A crossover transition might "
                              + "be possible if there is a link between the two.")
                    if phase2.key in phase1.low_trans:
                        transitions.append(
                            self.secondOrderTrans(V, phase1, phase2, verbose))
                    else:
                        pass
                    continue

                def DV(T):
                    return V(phase1.valAt(T), T) - V(phase2.valAt(T), T)

                DVmin, DVmax = DV(tmin), DV(tmax)

                if DVmin < 0:
                    # phase1 is lower at tmin, so no tunneling can occur
                    if verbose:
                        print("Tunneling cannot occur, because phase", i,
                              "is lower than phase", j, "at tmin: DV(tmin) =",
                              DVmin * self.conversionFactor**4, "GeV^4")
                    continue

                if DVmax > 0:
                    # phase1 is higher even at tmax. This can be weird, because a critical
                    # temperature in 1d potential needs to exist and be below Tmax. In 2d
                    # scenarios, however, this can happen without a problem. In the latter
                    # case, we need to make sure that this is not an unphysical transition
                    # 1 -> 0 though, from a phase 1 that will only become relevant at
                    # lower temperatures and might allow for 0 -> 1, assuming that the
                    # univere is in phase 0 at high temperatures. To check this, we
                    # require that the maximal temperature at which phase1 (the starting
                    # phase) is defined is higher than that of phase2 (the ending phase)
                    if phase1.Tmax > phase2.Tmax:
                        if verbose:
                            print("Possible transition without a critical temp. found." +
                                  "This can happen for multidimensional potentials.")
                        # Note: Tcrit here is not an actual critical temperature, but
                        # the largest temperature where the two phases coexist.
                        transition = TransitionInfo(
                            Tcrit=tmax,
                            high_vev=phase1.valAt(tmax),
                            high_phase=phase1.key,
                            low_vev=phase2.valAt(tmax),
                            low_phase=phase2.key,
                            type=1)
                        transitions.append(transition)
                    else:
                        if verbose:
                            print("Tunneling cannot occur, because phase", i,
                                  "is higher than phase", j, "at tmax: DV(tmax) =",
                                  DVmax * self.conversionFactor**4,
                                  "GeV^4 and the initial phase exists since higher temperatures.")
                    continue

                Tcrit = optimize.brentq(DV, tmin, tmax, disp=False)
                if verbose:
                    print("Found a possible transition from phase", i,
                          "to phase", j, "with Tcrit =",
                          Tcrit * self.conversionFactor, "GeV")
                transition = TransitionInfo(
                    Tcrit=Tcrit,
                    high_vev=phase1.valAt(Tcrit),
                    high_phase=phase1.key,
                    low_vev=phase2.valAt(Tcrit),
                    low_phase=phase2.key,
                    type=1)
                transitions.append(transition)

        if start_high:
            raise NotImplementedError("start_high=True not yet supported")

        return sorted(transitions, key=lambda x: x.Tcrit)[::-1]

    def getStartPhase(self, phases, V=None):
        """
        Find the key for the high-T phase.
        
        Parameters
        ----------
        phases : dict
            Output from :func:`traceMultiMin`.
        V : callable
            The potential V(x,T). Only necessary if there are
            multiple phases with the same Tmax.
        """
        startPhases = []
        startPhase = None
        Tmax = None
        assert len(phases) > 0
        for i in list(phases.keys()):
            if phases[i].T[-1] == Tmax:
                # add this to the startPhases list.
                startPhases.append(i)
            elif Tmax is None or phases[i].T[-1] > Tmax:
                startPhases = [i]
                Tmax = phases[i].T[-1]
        if len(startPhases) == 1 or V is None:
            startPhase = startPhases[0]
        else:
            # more than one phase have the same maximum temperature
            # Pick the stable one at high temp.
            Vmin = None
            for i in startPhases:
                V_ = V(phases[i].X[-1], phases[i].T[-1])
                if Vmin is None or V_ < Vmin:
                    Vmin = V_
                    startPhase = i
        assert startPhase in phases
        return startPhase

    def evalNucleationCriterion(self, T: float, start_phase: "PhaseInfo",
                                end_phase: "PhaseInfo", V: callable, dV: callable,
                                phitol: float, overlapAngle: float,
                                nuclCriterion: callable,
                                fullTunneling_params: dict, verbose: bool,
                                outdict: dict) -> float:
        """ Find the lowest action tunneling solution.

        This functions stores the results in the dict `outdict` with
        the temperature as the key.

        Parameters
        ----------
        T : float
            The temperature.
        start_phase : PhaseInfo
            The starting phase
        end_phase : PhaseInfo
            The end phase
        V : callable
            The effective potential
        dV : callable
            The gradient of the effective potential.

        Returns
        -------
        float :
            Evaluation of nuclCriterion(S,T)."""
        try:
            T = T[0]  # need this when the function is run from optimize.fmin
        except:
            pass

        # Loop through all the phases, adding acceptable minima
        x0 = start_phase.valAt(T)

        if T in outdict:
            return nuclCriterion(outdict[T]['action'], T, start_phase, end_phase)

        V0 = V(x0, T)
        p = end_phase

        if p.Tmin > T:
            if verbose:
                print("Phase %s not valid at T = %g" % (p.key, T * self.conversionFactor),
                      "GeV < Tmin = ", p.Tmin * self.conversionFactor, "GeV")
            return np.inf

        if p.Tmax < T:
            if verbose:
                print("Phase %s not valid at T = %g" % (p.key, T * self.conversionFactor),
                      "GeV > Tmax = ", p.Tmax * self.conversionFactor, "GeV")
            return -np.inf

        x1 = p.valAt(T)
        V1 = V(x1, T)
        if V1 >= V0:
            if verbose:
                print("Tunneling energetically not possible. Tmax might be too" +
                      " high or the numerical accuracy is not sufficient. This is not a problem, just a warning.")

        tdict = dict(low_vev=x1, high_vev=x0, Tnuc=T,
                     low_phase=end_phase.key, high_phase=start_phase.key)
        # Get rid of the T parameter for V and dV

        outdict = pathDeformation.bounceAction(
            T, V, dV, outdict, tdict, verbose=verbose,
            conversionFactor=self.conversionFactor,
            **fullTunneling_params)

        S = outdict[T]['action']
        return nuclCriterion(S, T, start_phase, end_phase)
