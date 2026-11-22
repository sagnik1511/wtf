# Optional same-species competition

[Worlds and species](worlds/index.md) · [Physics](physics.md) · [Rewards](rewards.md) · [Configuration](configuration.md)

Animals can learn a `challenge` action to compete with a same-species animal
nearby, including aggression without a resource objective. It is a policy decision, not an automatic response
to hunger or a blocked move. Existing presets keep their previous rules; this
feature is disabled unless explicitly configured.

![Challenge contexts, resolution and playback markers](assets/diagrams/competition.svg)

## Try the opt-in preset

[`competition.json`](../environments/competition.json) copies the maintained
masked-PPO setup and enables challenges for both species. It is a mechanics
example, not a calibrated improvement in population stability.

```sh
.venv/bin/python -m wtf.cli environments/competition.json --runs 1 --seed 100 --progress
```

To enable it in another environment, append `challenge` to the species'
`allowed_actions` and add this optional object to that same species:

```json
"competition": {
  "enabled": true,
  "damage": 5.0,
  "energy_cost": 1.0,
  "food": true,
  "blocked_path": true,
  "aggression": true
}
```

Omitting the object or setting `enabled: false` disables contests. Enabling
competition requires `challenge` in the action list. The switches for food,
blocked paths and aggression can be enabled independently. All three default to
true within enabled competition. For aggression alone, set `food: false` and
`blocked_path: false`, leaving `aggression: true`. Full field constraints are in
[the schema reference](configuration-reference.md).

## What is contested

| Situation | Eligibility | Winner's priority |
| --- | --- | --- |
| Shared resource food | Both animals can reach the same first available patch; the neighbour chooses `eat` or `challenge` | One normal bite before ordinary eaters; capacity, bite size, conversion and patch quantity still apply |
| Aggression | Any cardinally adjacent living same-species animal; no food or movement objective required | Configured damage only; no automatic feeding or displacement |
| Blocked path | A living same-species animal occupies the cell directly ahead | A winning challenger can advance after the blocker retreats into a legal free neighbouring cell; a winning defender stays in place |

Food contests apply to resource-eating species. Predators can challenge their
own species over blocked space or through aggression; hunting the same prey does not create a shared
resource-patch contest or an automatic fight. Different named species never
qualify as same-species opponents, even when they have the same diet.

Both participants must be on dry cells. Food/path contests target the cell
directly ahead. Aggression can also target side or rear neighbours: prefer an
eligible front neighbour, otherwise the adjacent animal with the smallest ID.
Targets are physical neighbours, not a species-wide effect. Food takes precedence when both food
intent and path blocking apply. If a food-only challenger's opponent rests,
turns or does something else, the food contest falls back to a path contest if enabled, then aggression if
enabled; otherwise it fails. Food and path objectives take precedence over
aggression for an eligible front neighbour.

## Resolution and costs

Normal movement and river drift happen first, then challenges resolve, followed
by feeding, predation, metabolism, deaths and reproduction. A potential target
can move away before the contest phase. Each animal participates in at most
one contest per step; reciprocal challenges produce one contest, not two hits.
When several animals challenge the same neighbour, existing animal iteration
order determines the first eligible pairing.

Contest strength is **current health / maximum health + current energy /
maximum energy**, after explicit challenge effort is charged. Higher strength
wins; exact ties use the world's seeded RNG. Each explicit challenge attempt
pays `energy_cost`, including failures and losses, in addition to ordinary
living demand. A defender does not pay challenge effort unless it also chose
`challenge`. Both parents' consent rules remain separate; choosing a challenge
cancels pending mating intent.

The winner deals up to `damage` actual health damage to the loser. A contest
can be lethal, but produces neither predation kill credit nor kill-energy
rewards. An animal killed in this phase cannot feed or attack afterward.

A food winner receives the first normal bite as part of resolving the contest;
it does not also get a second feeding turn. Other eaters may consume what is
left. A defending animal can win food priority without having chosen a challenge.

For a winning challenger to advance, the loser must retreat to an empty dry
adjacent cell, or have died in the contest. Retreat prefers the legal cell
furthest from the challenger, with coordinate ordering for ties. Obstacles,
occupancy and climb limits still apply; no swap into the challenger's old cell
is permitted. If no legal retreat exists, neither body moves, although the
contest damage still applies. Successful retreat and advance each charge their
own ordinary movement and uphill costs before metabolism. A defender that wins
keeps its cell; it does not automatically push its challenger.

## Learning, inspection and artifacts

All nine controllers support the action. Maskable PPO exposes it only when the
current physical opportunity exists; the mask does not read a neighbour's
forthcoming action or predict later movement. Other controllers can try an
unavailable challenge and receive ordinary failed-action feedback.

Existing reward coefficients apply: actual harm uses damage feedback, challenge
and displacement demand uses `per_energy_cost`, and an unsuccessful explicit
choice uses `failed_action`. A resolved losing challenge is a contest outcome,
not an additional failed-action penalty. No separate win bonus is added.
Positive damage rewards can encourage fighting, so review their balance with
energy demand and harm rather than assuming cooperation will emerge.

Replay events preserve attempts, resolved choices, winners/losers, damage,
food intake and explicit displacement positions. The viewer displays a contest
banner with both IDs, contest type, damage and event step. Amber rings around
both recorded fight locations and a connecting line distinguish fights from
hunting kills; the board labels the winner and health damage. Highlights last
12 recorded frames without pausing playback. They carry across sequential
chunks and retain at most 32 contests in memory, adding no replay storage.
The viewer includes `challenge` in action predictions and labels competition
deaths separately. The compact contest label overlays the board alongside the
kill/death label instead of adding a layout row, so event changes do not resize
the grid. Simultaneous top labels stack without covering each other. Overlay
colors follow the interface theme. Restart the playback server and
refresh the page after updating its readers and JavaScript.

Per-species behavior analysis and per-step telemetry expose contest attempts,
wins/losses, competition energy and deaths when the feature is enabled. Fighting
hits are excluded from hunting-hit rates; a challenger receiving a bite is not
counted as an ordinary `eat` decision. As with other events, recordings that skip
steps can omit contests; incomplete counts are not interpolated.

The configuration, body state and RNG already stored in checkpoints are enough
for exact continuation. No extra per-animal reservation or brain trace is saved;
contests finish within their simulation step. Existing replay formats remain
usable with updated readers. Adding `challenge` changes the ordered action
signature, so start fresh or load weights with the same action list. Exact
resume uses the checkpoint's saved configuration; editing a preset does not
retroactively enable contests in an old run.
