from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...components.memory import Memory
    from ...topology.node import Node

from numpy.random import random

from ..message import Message
from .entanglement_protocol import EntanglementProtocol
from ...topology.node import Node


class EntanglementSwappingMessage(Message):
    def __init__(self, msg_type: str, receiver: str, **kwargs):
        Message.__init__(self, msg_type, receiver)
        if self.msg_type == "SWAP_RES":
            self.fidelity = kwargs.get("fidelity")
            self.remote_node = kwargs.get("remote_node")
            self.remote_memo = kwargs.get("remote_memo")
        else:
            raise Exception("Entanglement swapping protocol create unkown type of message: %s" % str(msg_type))

    def __str__(self):
        if self.msg_type == "SWAP_RES":
            return "EntanglementSwappingMessage: msg_type: %s; local_memo: %d; fidelity: %.2f; " \
                   "remote_node: %s; remote_memo: %d; " % (self.msg_type, self.local_memo,
                                                           self.fidelity, self.remote_node,
                                                           self.remote_memo)


class EntanglementSwappingA(EntanglementProtocol):
    """
    Entanglement swapping protocol is an asymmetric protocol. The EntanglementSwappingA initiate protocol and do
    swapping operation. The EntanglementSwappingB waits swapping results from EntanglementSwappingA. The swapping
    results decides the following operations of EntanglementSwappingB.
    """

    def __init__(self, own: "Node", name: str, left_memo: "Memory", right_memo: "Memory", success_prob=1,
                 degradation=0.95):
        assert left_memo != right_memo
        EntanglementProtocol.__init__(self, own, name)
        self.memories = [left_memo, right_memo]
        self.left_memo = left_memo
        self.right_memo = right_memo
        self.success_prob = success_prob
        self.degradation = degradation
        self.is_success = False
        self.left_protocol = None
        self.right_protocol = None

    def is_ready(self) -> bool:
        return self.left_protocol is not None and self.right_protocol is not None

    def set_others(self, other: "EntanglementSwappingB") -> None:
        if other.own.name == self.left_memo.entangled_memory["node_id"]:
            self.left_protocol = other
        elif other.own.name == self.right_memo.entangled_memory["node_id"]:
            self.right_protocol = other
        else:
            raise Exception("Cannot pair protocol %s with %s" % (self.name, other.name))

    def start(self) -> None:
        assert self.left_memo.fidelity > 0 and self.right_memo.fidelity > 0
        assert self.left_memo.entangled_memory["node_id"] == self.left_protocol.own.name
        assert self.right_memo.entangled_memory["node_id"] == self.right_protocol.own.name

        fidelity = 0
        if random() < self.success_probability():
            fidelity = self.updated_fidelity(self.left_memo.fidelity, self.right_memo.fidelity)
            self.is_success = True

        msg = EntanglementSwappingMessage("SWAP_RES", self.left_protocol.name,
                                          fidelity=fidelity,
                                          remote_node=self.right_memo.entangled_memory["node_id"],
                                          remote_memo=self.right_memo.entangled_memory["memo_id"])
        self.own.send_message(self.left_protocol.own.name, msg)
        msg = EntanglementSwappingMessage("SWAP_RES", self.right_protocol.name,
                                          fidelity=fidelity,
                                          remote_node=self.left_memo.entangled_memory["node_id"],
                                          remote_memo=self.left_memo.entangled_memory["memo_id"])
        self.own.send_message(self.right_protocol.own.name, msg)

        self.left_memo.fidelity = self.right_memo.fidelity = 0
        self.left_memo.entangled_memory["node_id"] = self.right_memo.entangled_memory["node_id"] = None
        self.left_memo.entangled_memory["memo_id"] = self.right_memo.entangled_memory["memo_id"] = None
        self.update_resource_manager(self.left_memo, "RAW")
        self.update_resource_manager(self.right_memo, "RAW")

    def update_resource_manager(self, memory: "Memory", state: str) -> None:
        self.own.resource_manager.update(self, memory, state)

    def success_probability(self) -> float:
        '''
        A simple model for BSM success probability
        '''
        return self.success_prob

    def updated_fidelity(self, f1: float, f2: float) -> float:
        '''
        A simple model updating fidelity of entanglement
        '''
        return (f1 + f2) / 2 * self.degradation

    def received_message(self, src: str, msg: "Message") -> None:
        assert False


class EntanglementSwappingB(EntanglementProtocol):
    """
    Entanglement swapping protocol is an asymmetric protocol. The EntanglementSwappingA initiate protocol and do
    swapping operation. The EntanglementSwappingB waits swapping results from EntanglementSwappingA. The swapping
    results decides the following operations of EntanglementSwappingB.
    """

    def __init__(self, own: "Node", name: str, hold_memo: "Memory"):
        EntanglementProtocol.__init__(self, own, name)
        self.memories = [hold_memo]
        self.memory = hold_memo
        self.another = None

    def is_ready(self) -> bool:
        return self.another is not None

    def set_others(self, another: "EntanglementSwappingA") -> None:
        self.another = another

    def received_message(self, src: str, msg: "EntanglementSwappingMessage") -> None:
        assert src == self.another.own.name
        self.memory.fidelity = msg.fidelity
        if msg.fidelity > 0:
            self.memory.entangled_memory["node_id"] = msg.remote_node
            self.memory.entangled_memory["memo_id"] = msg.remote_memo
            self.update_resource_manager(self.memory, "ENTANGLED")
        else:
            self.memory.entangled_memory["node_id"] = None
            self.memory.entangled_memory["memo_id"] = None
            self.update_resource_manager(self.memory, "RAW")

    def update_resource_manager(self, memory: "Memory", state: str) -> None:
        self.own.resource_manager.update(self, memory, state)

    def start(self) -> None:
        pass
