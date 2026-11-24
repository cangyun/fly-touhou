"""Run identity and optimizer scope shared by the publication entry points."""

ENCODER_GROUPS = frozenset(("inherited_sensory_ports", "new_context_ports"))
NBNM_HEADS = ("movement", "risk", "native_warning48", "corridor_risk", "corridor_witness",
              "corridor_course", "corridor_endpoint")
CLEAR_HEADS = ("movement", "risk", "Bomb", "three_critics")
OUTPUT_GROUPS = frozenset((*NBNM_HEADS, *CLEAR_HEADS, "shared_directions", "three_critics"))


def trainable_groups(scope, *, objective, heads=None):
    if scope == "joint":
        if heads:
            raise ValueError("An explicit head list requires --train-scope encoder-heads")
        return None
    if scope != "encoder-heads":
        raise ValueError("Choose joint or encoder-heads")
    selected = frozenset(heads if heads is not None else (CLEAR_HEADS if objective == "clear" else NBNM_HEADS))
    if not selected or not selected <= OUTPUT_GROUPS:
        raise ValueError("Select one or more known output groups")
    if objective == "nbnm" and "Bomb" in selected:
        raise ValueError("NBNM does not update a Bomb head")
    return ENCODER_GROUPS | selected


def contract(game, difficulty, objective, scope, observations, *, heads=None):
    groups = trainable_groups(scope, objective=objective, heads=heads)
    return dict(schema="fly-touhou-run-v1", game=game, difficulty=difficulty, objective=objective,
                train_scope=scope, observation_version=observations,
                action_version="source19-native36-v1" if objective == "clear" else "source18-movement-v1",
                connectome_version="FlyWire-FAFB-v783",
                optimizer_groups=sorted(groups) if groups is not None else None)


def restore_contract(stored, expected):
    if expected is None:
        return
    saved = stored.get("run_contract") or stored.get("model_payload", {}).get("run_contract")
    if saved != expected:
        raise ValueError("Optimizer resume needs the same game, objective, training scope and interface. "
                         "Use the compatible parameters as --checkpoint to initialize a new optimizer.")


def mark_parameters(payload, identity):
    if identity is not None:
        observed = payload.get("input_schema")
        if observed != identity["observation_version"]:
            raise ValueError(f"The checkpoint uses {observed}, expected {identity['observation_version']}")
        payload["run_contract"] = identity
    return payload
