"""Entity-graph structure over live Razorpay webhook events.

WHY THIS TRANSFERS WHEN THE MODEL DOES NOT
-------------------------------------------
`core/demo_score.py` explains at length why the trained classifier cannot meaningfully
score a Razorpay payload: it wants 433 Vesta-engineered features and a webhook can supply
three.

The graph layer has no such problem. Connected components and k-core decomposition are
statements about topology. They make no distributional assumption, were fitted to nothing,
and do not care whether an identifier came from IEEE-CIS or from Razorpay — only whether
two entities share one. So the *algorithms* transfer to a new payment ecosystem intact,
even though the *model* does not.

That contrast is a genuine result about what carries across domains, and it is why the
dashboard shows both tracks side by side.

WHAT IS COMPUTED
----------------
Purely structural facts: how many other observed entities this one is linked to, the size
of its connected component, its core number, and which shared attribute did the linking.
No probability, no fraud claim, no threshold. A high core number here means "this entity is
densely connected to other entities that are themselves densely connected" — a factual
statement about the graph, not an accusation.

The algorithms are imported directly from `core/graph.py` — the same implementations
validated against networkx in `tests/test_graph.py`, not reimplementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.graph import MAX_GROUP_SIZE, connected_components, k_core_numbers

# Attributes that link two distinct payers. Sharing any one of these is a coincidence worth
# an edge; sharing several is what a ring looks like.
#
# Note the same hub-suppression discipline as the batch graph: a value held by more than
# MAX_GROUP_SIZE entities is generic (a popular email provider) rather than evidence of a
# relationship, and contributes no edges.
LINK_ATTRIBUTES = ("card_fingerprint", "email_domain", "contact", "vpa")


@dataclass
class LiveEvent:
    """The identifying surface of one payment, extracted from a webhook payload."""

    event_id: str
    payer_key: str
    card_fingerprint: str | None = None
    email_domain: str | None = None
    contact: str | None = None
    vpa: str | None = None

    def attributes(self) -> dict[str, str]:
        return {
            name: value
            for name, value in (
                ("card_fingerprint", self.card_fingerprint),
                ("email_domain", self.email_domain),
                ("contact", self.contact),
                ("vpa", self.vpa),
            )
            if value
        }


@dataclass
class StructuralResult:
    """What the graph can honestly say about one entity."""

    payer_key: str
    component_size: int
    core_number: int
    degree: int
    linked_payers: list[str] = field(default_factory=list)
    shared_attributes: list[str] = field(default_factory=list)
    total_entities: int = 0

    def to_dict(self) -> dict:
        return {
            "payer_key": self.payer_key,
            "component_size": self.component_size,
            "core_number": self.core_number,
            "degree": self.degree,
            "linked_payers": self.linked_payers,
            "shared_attributes": self.shared_attributes,
            "total_entities": self.total_entities,
        }


def extract_event(event_id: str, payment: dict) -> LiveEvent | None:
    """Pull Razorpay-native identifiers out of a payment entity.

    The payer key is the entity: the thing we are asking "is this linked to others?" about.
    Email is preferred, then phone, then UPI VPA, then the payment id as a last resort
    (which yields an entity linked to nothing, which is the correct answer when the payload
    carries no identifying surface at all).
    """
    card = payment.get("card") or {}
    fingerprint = None
    if isinstance(card, dict):
        parts = [card.get("last4"), card.get("network"), card.get("issuer")]
        if parts[0]:
            fingerprint = "|".join(str(p).lower() for p in parts if p)

    email = payment.get("email") or ""
    email_domain = email.split("@")[-1].lower() if "@" in email else None
    contact = str(payment.get("contact")).strip() if payment.get("contact") else None
    vpa = str(payment.get("vpa")).lower() if payment.get("vpa") else None

    payer_key = (
        (email.lower() if email else None)
        or contact
        or vpa
        or payment.get("id")
        or event_id
    )
    if not payer_key:
        return None

    return LiveEvent(
        event_id=event_id,
        payer_key=str(payer_key),
        card_fingerprint=fingerprint,
        email_domain=email_domain,
        contact=contact,
        vpa=vpa,
    )


def analyse(events: list[LiveEvent], target: LiveEvent) -> StructuralResult:
    """Build the bipartite entity graph over observed events and locate `target` in it.

    Node layout mirrors `core/graph.py`: entity nodes first, then attribute-value nodes.
    Bipartite rather than projected for the same reason as the batch graph — a group of k
    entities sharing a value costs k edges instead of k(k-1)/2, and connected components
    are identical either way.
    """
    payer_keys = sorted({event.payer_key for event in events} | {target.payer_key})
    index = {key: i for i, key in enumerate(payer_keys)}
    n_entities = len(payer_keys)

    # Collect every attribute value and which entities hold it.
    holders: dict[tuple[str, str], set[str]] = {}
    for event in [*events, target]:
        for name, value in event.attributes().items():
            holders.setdefault((name, value), set()).add(event.payer_key)

    adjacency: list[list[int]] = [[] for _ in range(n_entities)]
    next_node = n_entities
    shared: list[str] = []
    linked: set[str] = set()

    for (name, value), keys in sorted(holders.items()):
        if len(keys) < 2:
            continue  # a value held by one entity links nothing
        if len(keys) > MAX_GROUP_SIZE:
            continue  # hub: generic value, not evidence of a relationship

        attribute_node = next_node
        next_node += 1
        adjacency.append([])
        for key in sorted(keys):
            entity = index[key]
            adjacency[entity].append(attribute_node)
            adjacency[attribute_node].append(entity)

        if target.payer_key in keys:
            shared.append(name)
            linked.update(keys - {target.payer_key})

    labels = connected_components(adjacency)
    cores = k_core_numbers(adjacency)

    target_node = index[target.payer_key]
    target_label = labels[target_node]
    component_entities = sum(
        1 for i in range(n_entities) if labels[i] == target_label
    )

    return StructuralResult(
        payer_key=target.payer_key,
        component_size=component_entities,
        core_number=int(cores[target_node]),
        degree=len(adjacency[target_node]),
        linked_payers=sorted(linked),
        shared_attributes=sorted(set(shared)),
        total_entities=n_entities,
    )
