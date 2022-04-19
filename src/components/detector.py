"""Models for photon detection devices.

This module models a single photon detector (SPD) for measurement of individual photons.
It also defines a QSDetector class, which combines models of different hardware devices to measure photon states in different bases.
QSDetector is defined as an abstract template and as implementaions for polarization and time bin qubits.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List
from numpy import zeros, eye, kron, exp, sqrt
from scipy.linalg import fractional_matrix_power
from math import factorial

if TYPE_CHECKING:
    from ..kernel.timeline import Timeline

from ..components.photon import Photon
from ..components.beam_splitter import BeamSplitter, FockBeamSplitter
from ..components.switch import Switch
from ..components.interferometer import Interferometer
from ..components.circuit import Circuit
from ..kernel.entity import Entity
from ..kernel.event import Event
from ..kernel.process import Process
from ..kernel.quantum_utils import *
from ..utils.encoding import time_bin


class Detector(Entity):
    """Single photon detector device.

    This class models a single photon detector, for detecting photons.
    Can be attached to many different devices to enable different measurement options.

    Attributes:
        name (str): label for detector instance.
        timeline (Timeline): timeline for simulation.
        efficiency (float): probability to successfully measure an incoming photon.
        dark_count (float): average number of false positive detections per second.
        count_rate (float): maximum detection rate; defines detector cooldown time.
        time_resolution (int): minimum resolving power of photon arrival time (in ps).
        photon_counter (int): counts number of detection events.
    """

    def __init__(self, name: str, timeline: "Timeline", efficiency=0.9, dark_count=0, count_rate=int(25e6),
                 time_resolution=150):
        Entity.__init__(self, name, timeline)  # Detector is part of the QSDetector, and does not have its own name
        self.efficiency = efficiency
        self.dark_count = dark_count  # measured in 1/s
        self.count_rate = count_rate  # measured in Hz
        self.time_resolution = time_resolution  # measured in ps
        self.next_detection_time = -1
        self.photon_counter = 0

    def init(self):
        """Implementation of Entity interface (see base class)."""
        self.next_detection_time = -1
        self.photon_counter = 0
        if self.dark_count > 0:
            self.add_dark_count()

    def get(self, photon=None, **kwargs) -> None:
        """Method to receive a photon for measurement.

        Args:
            photon (Photon): photon to detect (currently unused)

        Side Effects:
            May notify upper entities of a detection event.
        """

        self.photon_counter += 1
        if self.get_generator().random() < self.efficiency:
            self.record_detection()

    def add_dark_count(self) -> None:
        """Method to schedule false positive detection events.

        Events are scheduled as a Poisson process.

        Side Effects:
            May schedule future `get` method calls.
            May schedule future calls to self.
        """

        assert self.dark_count > 0, "Detector().add_dark_count called with 0 dark count rate"
        time_to_next = int(self.get_generator().exponential(1 / self.dark_count) * 1e12)  # time to next dark count
        time = time_to_next + self.timeline.now()  # time of next dark count

        process1 = Process(self, "add_dark_count", [])  # schedule photon detection and dark count add in future
        process2 = Process(self, "record_detection", [])
        event1 = Event(time, process1)
        event2 = Event(time, process2)
        self.timeline.schedule(event1)
        self.timeline.schedule(event2)

    def record_detection(self):
        """Method to record a detection event.

        Will calculate if detection succeeds (by checking if we have passed `next_detection_time`)
        and will notify observers with the detection time (rounded to the nearest multiple of detection frequency).
        """

        now = self.timeline.now()

        if now > self.next_detection_time:
            time = round(now / self.time_resolution) * self.time_resolution
            self.notify({'time': time})
            self.next_detection_time = now + (1e12 / self.count_rate)  # period in ps

    def notify(self, info: Dict[str, Any]):
        """Custom notify function (calls `trigger` method)."""

        for observer in self._observers:
            observer.trigger(self, info)


class QSDetector(Entity, ABC):
    """Abstract QSDetector parent class.

    Provides a template for objects measuring qubits in different encoding schemes.

    Attributes:
        name (str): label for QSDetector instance.
        timeline (Timeline): timeline for simulation.
        detectors (List[Detector]): list of attached detectors.
        trigger_times (List[List[int]]): tracks simulation time of detection events for each detector.
    """

    def __init__(self, name: str, timeline: "Timeline"):
        Entity.__init__(self, name, timeline)
        self.detectors = []
        self.trigger_times = []

    def update_detector_params(self, detector_id: int, arg_name: str, value: Any) -> None:
        self.detectors[detector_id].__setattr__(arg_name, value)

    @abstractmethod
    def get(self, photon: "Photon", **kwargs) -> None:
        """Abstract method for receiving photons for measurement."""

        pass

    def trigger(self, detector: Detector, info: Dict[str, Any]) -> None:
        # TODO: rewrite
        detector_index = self.detectors.index(detector)
        self.trigger_times[detector_index].append(info['time'])
    
    def set_detector(self, idx: int,  efficiency=0.9, dark_count=0, count_rate=int(25e6), time_resolution=150):
        """Method to set the properties of an attached detector.

        Args:
            idx (int): the index of attached detector whose properties are going to be set.
            For other parameters see the `Detector` class. Default values are same as in `Detector` class.
        """
        assert idx >= 0 and idx <= len(self.detectors)-1, "`idx` must be a valid index of attached detector."

        detector = self.detectors[idx]
        detector.efficiency = efficiency
        detector.dark_count = dark_count
        detector.count_rate = count_rate
        detector.time_resolution = time_resolution

    def get_photon_times(self):
        return self.trigger_times

    @abstractmethod
    def set_basis_list(self, basis_list: List[int], start_time: int, frequency: float) -> None:
        pass


class QSDetectorPolarization(QSDetector):
    """QSDetector to measure polarization encoded qubits.

    There are two detectors.
    Detectors[0] and detectors[1] are directly connected to the beamsplitter.

    Attributes:
        name (str): label for QSDetector instance.
        timeline (Timeline): timeline for simulation.
        detectors (List[Detector]): list of attached detectors (length 2).
        trigger_times (List[List[int]]): tracks simulation time of detection events for each detector.
        splitter (BeamSplitter): internal beamsplitter object.
    """

    def __init__(self, name: str, timeline: "Timeline"):
        QSDetector.__init__(self, name, timeline)
        for i in range(2):
            d = Detector(name + ".detector" + str(i), timeline)
            self.detectors.append(d)
            d.attach(self)
        self.splitter = BeamSplitter(name + ".splitter", timeline)
        self.splitter.add_receiver(self.detectors[0])
        self.splitter.add_receiver(self.detectors[1])
        self.trigger_times = [[], []]

    def init(self) -> None:
        """Implementation of Entity interface (see base class)."""

        assert len(self.detectors) == 2

    def get(self, photon: "Photon", **kwargs) -> None:
        """Method to receive a photon for measurement.

        Forwards the photon to the internal polariaztion beamsplitter.

        Arguments:
            photon (Photon): photon to measure.

        Side Effects:
            Will call `get` method of attached beamsplitter.
        """

        self.splitter.get(photon)

    def get_photon_times(self):
        times = self.trigger_times
        self.trigger_times = [[], []]
        return times

    def set_basis_list(self, basis_list: List[int], start_time: int, frequency: float) -> None:
        self.splitter.set_basis_list(basis_list, start_time, frequency)

    def update_splitter_params(self, arg_name: str, value: Any) -> None:
        self.splitter.__setattr__(arg_name, value)


class QSDetectorTimeBin(QSDetector):
    """QSDetector to measure time bin encoded qubits.

    There are three detectors.
    The switch is connected to detectors[0] and the interferometer.
    The interferometer is connected to detectors[1] and detectors[2].

    Attributes:
        name (str): label for QSDetector instance.
        timeline (Timeline): timeline for simulation.
        detectors (List[Detector]): list of attached detectors (length 3).
        trigger_times (List[List[int]]): tracks simulation time of detection events for each detector.
        switch (Switch): internal optical switch component.
        interferometer (Interferometer): internal interferometer component.
    """

    def __init__(self, name: str, timeline: "Timeline"):
        QSDetector.__init__(self, name, timeline)
        self.switch = Switch(name + ".switch", timeline)
        self.detectors = [Detector(name + ".detector" + str(i), timeline) for i in range(3)]
        self.switch.add_receiver(self.detectors[0])
        self.interferometer = Interferometer(name + ".interferometer", timeline, time_bin["bin_separation"])
        self.interferometer.add_receiver(self.detectors[1])
        self.interferometer.add_receiver(self.detectors[2])
        self.switch.add_receiver(self.interferometer)

        self.components = [self.switch, self.interferometer] + self.detectors
        [component.attach(self) for component in self.components]
        self.trigger_times = [[], [], []]

    def init(self):
        """Implementation of Entity interface (see base class)."""

        pass

    def get(self, photon, **kwargs):
        """Method to receive a photon for measurement.

        Forwards the photon to the internal fiber switch.

        Args:
            photon (Photon): photon to measure.

        Side Effects:
            Will call `get` method of attached switch.
        """

        self.switch.get(photon)

    def get_photon_times(self):
        times, self.trigger_times = self.trigger_times, [[], [], []]
        return times

    def set_basis_list(self, basis_list: List[int], start_time: int, frequency: float) -> None:
        self.switch.set_basis_list(basis_list, start_time, frequency)

    def update_interferometer_params(self, arg_name: str, value: Any) -> None:
        self.interferometer.__setattr__(arg_name, value)


class QSDetectorFockDirect(QSDetector):
    """QSDetector with two input ports and two photon detectors which directly measure input photons' Fock states, respectively.

    Usage: to measure diagonal elements of effective density matrix.

    Attributes:
        name (str): label for QSDetector instance.
        timeline (Timeline): timeline for simulation.
        src_list (List[str]): list of two sources which send photons to this detector (length 2).
        detectors (List[Detector]): list of attached detectors (length 2).
        trigger_times (List[List[int]]): tracks simulation time of detection events for each detector.
        arrival_times (List[List[int]]): tracks simulation time of Photon arrival at each input port
    """

    def __init__(self, name: str, timeline: "Timeline", src_list: List[str]):
        super().__init__(name, timeline)
        assert len(src_list) == 2
        self.src_list = src_list
        for i in range(2):
            d = Detector(name + ".detector" + str(i), timeline)
            self.detectors.append(d)
            d.attach(self)
        self.trigger_times = [[], []]
        self.arrival_times = [[], []]

    def init(self):
        pass

    def _generate_povms(self):
        """Method to generate POVM operators corresponding to photon detector having 0 and 1 click
        Will be used to generated outcome probability distribution.
        """
        # assume using Fock quantum manager
        truncation = self.timeline.quantum_manager.truncation
        create, destroy = self.timeline.quantum_manager.build_ladder()
        series_elem_list = [(-1)**i*fractional_matrix_power(create,i+1).dot(fractional_matrix_power(destroy,i+1))/factorial(i+1) for i in range(truncation)]
        povm1 = sum(series_elem_list)
        povm0 = zeros((truncation+1, truncation+1)) - povm1

        return povm0, povm1

    def get(self, photon: "Photon", **kwargs):
        src = kwargs["src"]
        assert photon.encoding_type["name"] == "fock", "Photon must be in Fock representation."
        input_port = self.src_list.index[src] # determine at which input the Photon arrives, an index
        # record arrival time
        arrival_time = self.timeline.now()
        self.arrival_times[input_port].append(arrival_time)

        truncation = self.timeline.quantum_manager.truncation
        key = photon.quantum_state # the photon's key pointing to the quantum state in quantum manager
        state = self.timeline.quantum_manager.states[key].state # the quantum state as array
        keys = self.timeline.quantum_manager.states[key].keys # the list of all keys pointing to the quantum state in quantum manager
        num_sys = len(keys) # total number of subsystems of the (entangled) composite system
        povms = tuple(self._generate_povms()) # generate a tuple of two POVM operators
        idx = keys.index(key) # the relative index of the subsystem to be measured, needed if using cached measurement function in quantum_utils.py
        result = measure_entangled_state_with_cache_fock_density(tuple(state), idx, num_sys, povms, truncation = truncation) # get results
        
        # determine outcome
        prob0, prob1 = result[1] # probabilities for outcomes, prob1 corresponds to having a click
        if self.get_generator().random() <= prob1:
            detector_num = self.src_list.index(src)
            self.detectors[detector_num].get()
            # record trigger time
            trigger_time = self.timeline.now()
            self.trigger_times[detector_num].append(trigger_time)
            state = array(result[0][1]) # post measurement state
        else:
            state = array(result[0][0]) # post measurement state
        
        # trace out the subsystem corresponding to `key` and update the recorded state in quantum manager
        state_partial_trace = density_partial_trace(tuple(state), (idx,), num_sys, truncation=truncation)
        self.timeline.quantum_manager.states[key] = None # clear the stored state at `key`
        keys.remove(key)
        self.timeline.quantum_manager.set(keys, state_partial_trace)
        for i in keys:
            self.timeline.quantum_manager.states[i].keys.remove(key)

    def get_photon_times(self) -> List[List[int]]:
        trigger_times = self.trigger_times
        self.trigger_times = [[], []]
        return trigger_times

    # does nothing for this class
    def set_basis_list(self, basis_list: List[int], start_time: int, frequency: int) -> None:
        pass


class QSDetectorFockInterference(QSDetector):
    """QSDetector with two input ports and two photon detectors behind beamsplitter which physically measure two beamsplitter output 
        photonic modes' Fock states, respectively. POVM operators which apply to pre-beamsplitter photonic state are used.
        NOTE: in current implementation, to realize interference we require Photons arrive at both input ports simultaneously,
        and at most 1 Photon instance can be input at an input port at a time.

    Usage: to realize Bell state measurement (BSM) and to measure off-diagonal elements of effective density matrix.

    Attributes:
        name (str): label for QSDetector instance.
        timeline (Timeline): timeline for simulation.
        src_list (List[str]): list of two sources which send photons to this detector (length 2).
        detectors (List[Detector]): list of attached detectors (length 2).
        trigger_times (List[List[int]]): tracks simulation time of detection events for each detector.
        phase (float): relative phase between two input optical paths.
        temporary_photon_info (List[Dict]): temporary list of information of Photon arriving at each input port. 
            Specific to current implementation. At most 1 Photon's information will be recorded in a dictionary. 
            When there are 2 non-empty dictionaries, e.g. [{"photon":Photon1, "time":arrival_time1}, {"photon":Photon2, "time":arrival_time2}],
            the entangling measurement will be carried out. After measurement the temporary list will be reset.
    """

    def __init__(self, name: str, timeline: "Timeline", src_list: List[str], phase: float=0):
        super().__init__(name, timeline)
        assert len(src_list) == 2
        self.src_list = src_list
        self.phase = phase
        # self.beamsplitter = FockBeamSplitter(name + ".beamsplitter", timeline)
        for i in range(2):
            d = Detector(name + ".detector" + str(i), timeline)
            self.beamsplitter.add_receiver(d)
            self.detectors.append(d)
            d.attach(self)

        # default circuit: no phase applied
        # self._circuit = Circuit(2)

        self.trigger_times = [[], []]
        self.arrival_times = [[], []]
        self.temporary_photon_info = [{}, {}]

    def init(self):
        pass

    def _generate_transformed_ladders(self):
        """Method to generate transformed creation/annihilation operators by the beamsplitter.
        Will be used to construct POVM operators.
        """

        identity = eye(self.truncation+1)
        create, destroy = self.timeline.quantum_manager.build_ladder()
        phase = self.phase
        efficiency1 = self.detectors[0].efficiency
        efficiency2 = self.detectors[1].efficiency

        # Modified mode operators in Heisenberg picture by beamsplitter transformation considering inefficiency (ignoring relative phase)
        create1 = (kron(efficiency1*create,identity) + exp(1j*phase)*kron(identity,efficiency2*create))/sqrt(2)
        destroy1 = create1.conj().T
        create2 = (kron(efficiency1*create,identity) - exp(1j*phase)*kron(identity,efficiency2*create))/sqrt(2)
        destroy2 = create2.conj().T

        return create1, destroy1, create2, destroy2

    def _generate_povms(self):
        """Method to generate POVM operators corresponding to photon detector having 00, 01, 10 and 11 click(s)
        Will be used to generated outcome probability distribution.
        """
        # assume using Fock quantum manager
        truncation = self.timeline.quantum_manager.truncation
        create1, destroy1, create2, destroy2 = self._generate_transformed_ladders()
        # for detector1 (index 0)
        series_elem_list1 = [(-1)**i*fractional_matrix_power(create1,i+1).dot(fractional_matrix_power(destroy1,i+1))/factorial(i+1) for i in range(truncation)]
        povm1_1 = sum(series_elem_list1)
        povm0_1 = zeros(((truncation+1)**2, (truncation+1)**2)) - povm1_1
        # for detector2 (index 1)
        series_elem_list2 = [(-1)**i*fractional_matrix_power(create2,i+1).dot(fractional_matrix_power(destroy2,i+1))/factorial(i+1) for i in range(truncation)]
        povm1_2 = sum(series_elem_list2)
        povm0_2 = zeros(((truncation+1)**2, (truncation+1)**2)) - povm1_2

        # POVM operators for 4 possible outcomes
        # Note: povm01 and povm10 are relevant to BSM
        povm00 = povm0_1 @ povm0_2
        povm01 = povm0_1 @ povm1_2
        povm10 = povm1_1 @ povm0_2
        povm11 = povm1_1 @ povm1_2

        return povm00, povm01, povm10, povm11

    def get(self, photon, **kwargs):
        src = kwargs["src"]
        assert photon.encoding_type["name"] == "fock", "Photon must be in Fock representation."
        input_port = self.src_list.index[src] # determine at which input the Photon arrives, an index
        # record arrival time
        arrival_time = self.timeline.now()
        self.arrival_times[input_port].append(arrival_time)
        # record in temporary photon list
        assert not self.temporary_photon_info[input_port], "At most 1 Photon instance should arrive at an input port at a time."
        self.temporary_photon_info[input_port]["photon"] = photon
        self.temporary_photon_info[input_port]["time"] = arrival_time

        # judge if there have already been two input Photons arrving simultaneously
        dict0 = self.temporary_photon_info[0]
        dict1 = self.temporary_photon_info[1]
        # if both two dictionaries are non-empty
        if dict0 and dict1:
            assert dict0["time"] == dict1["time"], "To realize interference photons must arrive simultaneously."
            photon0 = dict0["photon"]
            photon1 = dict1["photon"]
            key0 = photon0.quantum_state
            key1 = photon1.quantum_state
            keys0 = self.timeline.quantum_manager.states[key0].keys
            keys1 = self.timeline.quantum_manager.states[key1].keys
            idx0 = keys0.index(key0)
            idx1 = keys1.index(key1)
            state0 = self.timeline.quantum_manager.states[key0].state
            state1 = self.timeline.quantum_manager.states[key1].state
            composite_state = tuple(kron(state0, state1)) # note the order in tensor (kronecker) product to generate the composite system
            keys = keys0 + keys1 # all keys pointing to the new composite system
            indices = (idx0, idx1+len(keys0))
            num_sys = len(keys0) + len(keys1)
            povms = tuple(self._generate_povms())
            truncation = self.timeline.quantum_manager.truncation

            # determine the outcome
            result = measure_multiple_with_cache_fock_density(composite_state, indices, num_sys, povms, truncation=truncation)
            prob_dist = array(result[1]) # probability distribution for outcomes
            outcomes = ["00", "01", "10", "11"]
            outcome = self.get_generator().choice(outcomes, p=prob_dist) # random choice to determine outcome
            if outcome == "00":
                # no click at all
                state = array(result[0][0]) # post measurement state

            elif outcome == "01":
                # detector 1 has a click
                self.detectors[1].get()
                # record trigger time
                trigger_time = self.timeline.now()
                self.trigger_times[1].append(trigger_time)
                state = array(result[0][1]) # post measurement state

            elif outcome == "10":
                # detector 0 has a click
                self.detectors[0].get()
                # record trigger time
                trigger_time = self.timeline.now()
                self.trigger_times[0].append(trigger_time)
                state = array(result[0][2]) # post measurement state

            else:
                # both detectors have a click
                self.detectors[0].get()
                self.detectors[1].get()
                # record trigger time
                trigger_time = self.timeline.now()
                self.trigger_times[0].append(trigger_time)
                self.trigger_times[1].append(trigger_time)
                state = array(result[0][3]) # post measurement state

            # trace out the subsystem corresponding to `key` and update the recorded state in quantum manager
            state_partial_trace = density_partial_trace(tuple(state), (idx0,idx1), num_sys, truncation=truncation)
            self.timeline.quantum_manager.states[key0] = None # clear the stored state at `key0`
            self.timeline.quantum_manager.states[key1] = None # clear the stored state at `key1`
            keys.remove([key0,key1])
            self.timeline.quantum_manager.set(keys, state_partial_trace)
            for i in keys:
                self.timeline.quantum_manager.states[i].keys = keys

        else:
            pass

        """
        # check if we have non-null photon
        if not photon.is_null:
            state = self.timeline.quantum_manager.get(photon.quantum_state)

            # if entangled, apply phase gate
            if len(state.keys) == 2:
                self.timeline.quantum_manager.run_circuit(self._circuit, state.keys)

            self.beamsplitter.get(photon)
        """

    def get_photon_times(self) -> List[List[int]]:
        trigger_times = self.trigger_times
        self.trigger_times = [[], []]
        return trigger_times

    # does nothing for this class
    def set_basis_list(self, basis_list: List[int], start_time: int, frequency: float) -> None:
        pass

    def set_phase(self, phase: float):
        self.phase = phase
        """
        self._circuit = Circuit(2)
        self._circuit.phase(0, phase)
        """