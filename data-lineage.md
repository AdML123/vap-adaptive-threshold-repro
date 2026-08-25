# Data lineage and release boundary

| Resource | Role | Release status |
|---|---|---|
| Switchboard-1 Release 2 audio | model input | Restricted; not redistributed |
| ISIP/MSU word alignments | proxy VAD source | Restricted/external; not redistributed |
| SWBD-DAMSL annotations | backchannel labels | Restricted/external; not redistributed |
| Official VAP checkpoint | frozen backbone | External artifact; not redistributed |
| Per-bin loss tables | aggregate derived output | Included |
| Head-only test metrics | aggregate derived output | Included |
| Figures 1 and 2 | disclosure-safe derived outputs | Included |
| Training/evaluation scripts | source code | Included |

The package is reproducible in the sense that an authorized user can provide
the omitted inputs locally and follow the documented commands. It does not grant
permission to redistribute the restricted corpus or checkpoint.
