# Game Server Deep Dive — Research Notes

Compiled 2026-03-15. Production numbers, war stories, specific implementations.

---

## 1. Game Loop / Tick Rate

### The Fixed Timestep Accumulator Pattern

The canonical reference is Glenn Fiedler's "Fix Your Timestep!" (gafferongames.com).

```
const TICK_RATE: f64 = 60.0;
const DT: f64 = 1.0 / TICK_RATE;  // 16.67ms per tick

let mut accumulator: f64 = 0.0;
let mut previous_time = now();

loop {
    let current_time = now();
    let frame_time = (current_time - previous_time).min(0.25); // clamp spiral-of-death
    previous_time = current_time;
    accumulator += frame_time;

    while accumulator >= DT {
        simulate(state, DT);   // physics/game logic at fixed rate
        accumulator -= DT;
    }

    let alpha = accumulator / DT;  // 0.0..1.0
    render(interpolate(prev_state, state, alpha)); // smooth rendering
}
```

Key insight: the `0.25` clamp prevents the "spiral of death" where a slow frame causes
multiple catch-up ticks, which cause more slow frames. If the server falls behind by more
than 250ms, it drops ticks rather than trying to catch up.

### Production Tick Rates

| Game              | Tick Rate   | Notes                                          |
|-------------------|------------|------------------------------------------------|
| Overwatch (1/2)   | 63 Hz      | Client update was only 20Hz at launch, bumped to 63Hz |
| CS:GO             | 64 Hz      | Official MM; FACEIT/ESEA ran 128 Hz            |
| CS2               | 64 Hz      | Sub-tick system timestamps actions between ticks |
| Valorant          | 128 Hz     | Default MM, a key marketing differentiator      |
| Fortnite          | 30 Hz      | Battle royale scale tradeoff                   |
| Apex Legends      | 20 Hz      | 60 players, server cost tradeoff               |
| Minecraft         | 20 TPS     | 50ms per tick; redstone tick = 100ms (2 game ticks) |
| EVE Online        | 1 Hz       | Single-shard MMO; drops to 0.1 Hz under TiDi   |
| RuneScape         | 1.67 Hz    | 600ms tick, legacy from 2001                   |

### Why 20-60 Hz Is Typical

- **16.67ms budget at 60Hz** — most server ticks must complete in this window
- **Bandwidth**: at 60Hz with 64 players, even small snapshots (500B) = 1.9 MB/s outbound
- **Diminishing returns**: human reaction time ~200ms; beyond 60Hz, perceptual gains are marginal
  for most genres (exception: competitive FPS at 128Hz)
- **Server cost**: doubling tick rate roughly doubles CPU cost; cloud game servers at scale
  run 10-50 instances per physical machine

### Server-Authoritative Model

The server is the single source of truth. Clients send inputs (movement commands, actions),
the server simulates, and broadcasts authoritative state. This prevents most cheats by
design — the client never decides "I hit you", only "I pressed fire at time T."

### CS2 Sub-Tick Innovation

CS2 replaced the traditional tick-rate debate with sub-tick: player actions (shots, movement)
carry precise timestamps between ticks. The server interpolates these to determine exact
event ordering. Valve forced all third-party servers to 64Hz to unify the experience.
Community reaction was mixed — many competitive players still wanted 128Hz raw tick rate.

---

## 2. UDP for Games

### TCP Head-of-Line Blocking — The Core Problem

TCP guarantees in-order delivery. If packet #47 is lost in a stream of [47, 48, 49, 50],
packets 48-50 sit in the kernel receive buffer, blocked, until 47 is retransmitted and
arrives. At 60Hz with 50ms RTT, a single lost packet stalls ~3 ticks of game state.

For game state that is replaced every tick, this is catastrophic — you're waiting for stale
data that will be overwritten anyway.

### Building Reliability on UDP

Games typically implement 2-3 "channels" over raw UDP:

```
// Channel types:
// 1. Unreliable       — fire-and-forget (position updates, voice)
// 2. Reliable ordered — TCP-like, for chat, inventory, RPC
// 3. Reliable unordered — events that must arrive but order doesn't matter

struct PacketHeader {
    sequence: u16,          // wrapping sequence number
    ack: u16,               // last received remote sequence
    ack_bitfield: u32,      // bitmask of 32 previous acks (ack-1 through ack-32)
    channel: u8,
}
```

The ack_bitfield trick (from Gaffer on Games): a single 32-bit field encodes which of the
last 32 packets were received. Bit 0 = ack-1, bit 1 = ack-2, etc. Sender checks which
packets were NOT acked after ~100ms and retransmits only those (selective retransmission).

### ENet Library

- C library, ~4k lines, used by many indie and mid-tier games
- Provides reliable/unreliable channels, fragmentation/reassembly, connection management
- Packet aggregation: combines multiple small messages into one UDP datagram
- Built-in bandwidth throttling and flow control
- No encryption (layer TLS/DTLS on top)

### KCP Protocol

- By skywind3000, popular in Chinese game industry
- Trades 10-20% more bandwidth for 30-40% lower latency vs TCP
- Configurable: `nodelay`, `interval`, `resend`, `nc` (no congestion window)
- Aggressive retransmission: can retransmit before RTO via "fast resend" (2 duplicate acks)
- ARQ (Automatic Repeat reQuest) at application layer, runs over raw UDP
- Used in: Genshin Impact (miHoYo), many Chinese mobile MMOs

### QUIC as Middle Ground

- Stream multiplexing without HOL blocking between streams (lost packet only blocks its stream)
- Built-in TLS 1.3 (0-RTT connection establishment)
- Connection migration (survives IP changes — good for mobile)
- Downside for games: still has per-stream ordering and congestion control that may be
  too conservative for real-time game state
- Best for: game services (auth, matchmaking, chat) rather than real-time gameplay

---

## 3. Entity Component System (ECS)

### Why ECS for Game Servers

Traditional OOP (deep inheritance hierarchies) creates:
- Cache misses: `Player` object has 200 fields, iterating "all positions" touches scattered memory
- Tight coupling: changing `Enemy` base class breaks `FlyingEnemy`, `BossEnemy`, etc.

ECS separates data (Components) from behavior (Systems) and identity (Entities).

### Struct of Arrays vs Array of Structs

```
// Array of Structs (OOP/traditional) — poor cache behavior for iteration
struct Entity { Position pos; Velocity vel; Health hp; Inventory inv; ... }
Entity entities[10000]; // iterating positions touches 200-byte stride

// Struct of Arrays (ECS) — cache-friendly iteration
Position positions[10000];   // contiguous, 12 bytes each
Velocity velocities[10000];  // contiguous, 12 bytes each
// A system iterating positions streams through a tight array
```

A 64-byte cache line fits ~5 `Position` structs (vec3 = 12 bytes). With SoA layout, iterating
10k positions is ~24KB — fits in L1 cache. With AoS and 200-byte entities, same iteration
touches ~195KB, blowing L1 and thrashing L2.

### Archetype Storage (Bevy, Unity DOTS)

Entities with the same component set are stored in the same "archetype table." Each column
is one component type, backed by a contiguous `BlobVec`. Adding/removing components moves
the entity to a different archetype table.

**Bevy specifics:**
- Two storage strategies: **Table** (default, fast iteration, slow add/remove) and
  **SparseSet** (fast add/remove, slower iteration)
- Systems declare dependencies via parameter types: `Query<(&Position, &mut Velocity), With<Enemy>>`
- Automatic parallel scheduling when systems don't conflict
- Archetype fragmentation warning: too many unique component combos = too many small tables

**Unity DOTS specifics:**
- `IComponentData` for unmanaged structs, stored in chunks (16KB each)
- `SystemBase` / `ISystem` for Burst-compiled jobs
- `EntityQuery` with change filters (only process entities whose component changed this frame)
- `NativeArray<T>` with safety checks for parallel job access

### Spatial Structures for Interest Management

ECS alone doesn't solve spatial queries. Common patterns:

**Spatial Hashing (most common for game servers):**
```
// Divide world into cells (e.g., 100x100 units)
cell_x = floor(position.x / CELL_SIZE)
cell_y = floor(position.z / CELL_SIZE)
hash = cell_x * PRIME + cell_y
// HashMap<hash, Vec<EntityId>>
// Rebuild every tick (cheap at 10k entities, ~0.5ms)
```

**Quadtree/Octree**: better for non-uniform distributions but harder to update incrementally.
Most game servers prefer spatial hashing because rebuild-per-tick is simpler and fast enough.

**Area-of-Interest (AOI) Filtering:**
- Each player has an interest radius (e.g., 200 units)
- Server only sends entity updates for entities within that radius
- Reduces bandwidth from O(n^2) to O(n * k) where k = nearby entities
- Photon, Mirror, SpatialOS all implement this

---

## 4. Netcode / Prediction

### Client-Side Prediction (Source Engine Approach)

```
// Client:
1. Read input (move forward)
2. Predict locally: apply movement physics, render immediately
3. Stamp input with sequence number, send to server
4. Store: pending_inputs[seq] = { input, predicted_state }

// Server:
1. Receive input with sequence number
2. Apply to authoritative simulation
3. Send back: { seq: last_processed_input, authoritative_state }

// Client reconciliation:
1. Receive server state for seq N
2. Discard pending_inputs[0..N] (server confirmed these)
3. Compare server state with predicted state at seq N
4. If mismatch: snap to server state, re-apply pending_inputs[N+1..current]
   (this is the "rewind and replay" step)
```

The re-application step is why prediction needs deterministic physics — same inputs must
produce same outputs. Non-determinism (floating point order, random seeds) causes constant
correction jitter.

### Entity Interpolation

Other players are rendered in the past, not the present:

```
// Interpolation buffer: store last N server snapshots with timestamps
// Render time = server_time - INTERP_DELAY (typically 100ms = 2 snapshots at 20Hz)

// Source engine defaults:
//   cl_interp     = 0.1    (100ms interpolation delay)
//   cl_updaterate = 20     (snapshots per second from server)
//   cl_cmdrate    = 30     (inputs per second to server)
//   cl_interp_ratio = 2    (buffer 2 snapshots minimum)

// At 66-tick Source servers:
//   cl_interp can drop to 0.0152 (1/66) with cl_updaterate 66
//   Practical minimum: ~30ms with good connection

fn interpolate(buffer: &[Snapshot], render_time: f64) -> State {
    // Find two snapshots bracketing render_time
    let (a, b) = find_bracket(buffer, render_time);
    let t = (render_time - a.time) / (b.time - a.time);
    lerp(a.state, b.state, t)
}
```

**Jitter buffer**: if snapshots arrive with variable timing (jitter), the interpolation
buffer absorbs this by always staying 2+ snapshots behind. More jitter = larger buffer =
more visual delay. Adaptive jitter buffers grow/shrink based on measured packet timing.

### Lag Compensation — Server-Side Rewind

**Overwatch "Favor the Shooter":**
- Uses client-side hit detection: client says "I hit player X in the head"
- Server validates: was there a clean line of sight at the time the client fired?
- Server rewinds other players' positions to where they were at the client's fire time
- Compensation window: typically capped at ~200-250ms (reject hits from >250ms ago)
- Exception: defensive abilities (Zarya bubble, Reaper wraith) are NOT rewound — they
  activate on the server timeline, creating the "I was already behind the wall" feeling

**Source Engine Lag Compensation:**
```
// Server stores position history for all players: 1 second of history
// sv_maxunlag = 1.0 (max seconds of rewind)

// When processing a shot:
1. Compute client's command execution time:
   cmd_time = current_server_time - client_latency - cl_interp
2. For each player, find their position at cmd_time from history buffer
3. Move hitboxes back to those positions
4. Perform ray/hitbox intersection
5. Restore all positions to present
6. Apply damage if hit
```

**CS:GO interp settings (competitive):**
- `cl_interp 0` with `cl_interp_ratio 1` at 128-tick = 7.8ms interp delay
- `cl_interp 0` with `cl_interp_ratio 2` at 64-tick = 31.25ms interp delay
- Total lag compensation window = client latency + interp delay
- `sv_maxunlag 1.0` — server won't rewind more than 1 second

### Delta Compression

**Quake 3 Delta Snapshots (the gold standard):**
```
// Server keeps last 32 snapshots per client in a ring buffer
// Client ACKs: "I received snapshot #47"
// Server computes delta: snapshot_current XOR snapshot_47

// Per-field encoding:
for each field in entity_state:
    if field == baseline_field:
        write_bit(0)           // unchanged, 1 bit
    else:
        write_bit(1)           // changed
        write_bits(field, field_bit_width)

// Entity presence:
for each entity:
    if entity in current but not baseline:
        write: ENTITY_ADD(entity_id, full_state)
    if entity in baseline but not current:
        write: ENTITY_REMOVE(entity_id)
    if entity in both:
        write: ENTITY_DELTA(entity_id, changed_fields_only)
```

- Huffman compression on top (fixed tree, not adaptive per-packet)
- History buffer is THE key innovation: allows delta encoding over unreliable UDP
  (if a packet is lost, the server just deltas against the last ACKed snapshot)
- Typical compression: 10-50x reduction vs full snapshots

---

## 5. Essential Game Networking Talks & References

### "Overwatch Gameplay Architecture and Netcode" — Timothy Ford, GDC 2017
- GDC Vault: gdcvault.com/play/1024001/
- YouTube: youtube.com/watch?v=W3aieHjyNvw
- Key takeaways:
  - ECS architecture with ~40 component types
  - Deterministic simulation: same inputs → same outputs across client/server
  - Prediction: client runs ahead by half-RTT + 1 command frame
  - Rollback: on mismatch, rewind to last confirmed state, re-simulate
  - `UpdateFixed` system — executes for every fixed command frame
  - Netcode designed around ECS: systems can be replayed cheaply because
    components are just data arrays

### "I Shot You First: Networking the Gameplay of Halo: Reach" — David Aldridge, GDC 2011
- GDC Vault: gdcvault.com/play/1014345/
- Key takeaways:
  - Hosted client model (one of 16 players acts as server)
  - Target-relative gunshot information: "I shot at player X's head" not "I shot at world pos Y"
  - Server validates line-of-sight, not raw position
  - Network profiling built into replay system — analyze bandwidth per player per frame
  - Reduced bandwidth to 20% of Halo 3 using profiling-driven optimization
  - Prediction for lag hiding: change client state before server confirmation

### Valve's "Source Multiplayer Networking" Wiki
- developer.valvesoftware.com/wiki/Source_Multiplayer_Networking
- THE canonical reference for client prediction + entity interpolation + lag compensation
- Covers: cl_interp, cl_updaterate, cl_cmdrate, tick rate, user command processing,
  data compression, lag compensation algorithm with position history buffer

### Glenn Fiedler's "Gaffer On Games" (gafferongames.com)
- **Fix Your Timestep!** — accumulator pattern, spiral of death, interpolated rendering
- **Networking for Game Programmers** series:
  - Sending/receiving packets, virtual connections over UDP
  - Reliability and flow control, sequence numbers and ack bitfields
- **Snapshot Interpolation** — linear interpolation between server snapshots
- **Snapshot Compression** — delta encoding, bit-packing, quantization
- **State Synchronization** — deterministic lockstep vs snapshot interpolation vs state sync
- **Reading and Writing Packets** — bit-packing implementation (read/write N bits)

### Gabriel Gambetta's "Fast-Paced Multiplayer" Series
- gabrielgambetta.com/client-side-prediction-server-reconciliation.html
- 4-part series with interactive live demos
- Clearest visual explanation of prediction + reconciliation + interpolation

---

## 6. Production Game Server Architectures

### Photon Server (Exit Games)
- **Model**: Room-based; Master server assigns players to GameServers
- **Scale**: up to 5,000 CCU per cloud server; most games use 2-16 players per room
- **Limit**: 500 messages/second per room (hard limit on Photon Cloud)
- **Interest management**: required for >32 players in fast-paced games
- **Pricing**: free up to 20 CCU; 1,000 CCU = $185/month; CCU burst (auto-scale) included for 500+ plans
- **Transports**: UDP (default), TCP, WebSocket
- **Used by**: many Unity games; room-based model fits lobby shooters, card games, party games

### Mirror (Unity Open Source)
- **Model**: Server+Client in one codebase; dedicated or host-client
- **Transport**: pluggable — Telepathy (TCP), KCP (UDP), WebSocket, Steam
- **API**: `[Server]`/`[Client]` tags, `[Command]` (C→S), `[ClientRpc]` (S→C), `[SyncVar]`
- **Spatial**: built-in Spatial Hashing & Distance Checker for AOI
- **Scale**: built/tested for MMO scale by uMMORPG team; uMMORPG = <6,000 LOC
- **Used by**: Population: ONE (Meta Quest), 1,000+ Steam games
- **Note**: fork of UNET; community maintained, no corporate backing

### Nakama (Heroic Labs)
- **Model**: monolith server (all features in one binary), written in Go
- **DB**: CockroachDB or PostgreSQL
- **Scripting**: Go, TypeScript, or Lua for server-side game logic
- **Features**: accounts, social, chat, matchmaking, leaderboards, storage, IAP validation
- **Multiplayer**: relayed (client-authoritative) and server-authoritative modes
- **Scale**: tested to 2 million CCU
- **Matchmaking**: flexible — match players to matches or players to players with custom properties
- **Protocol**: gRPC + gRPC-Gateway (HTTP/REST), Protocol Buffers, WebSocket for realtime
- **Used by**: mobile games, social games; enterprise tier for larger studios
- **Open source**: Apache 2.0 license

### SpatialOS (Improbable)
- **Model**: Entity-Component-Worker architecture on distributed cloud compute
- **Key innovation**: "workers" — micro-services that each own a region of the world
  - Workers overlap and dynamically reorganize as players move
  - A single entity can be managed by multiple workers
  - Seamless world: no zone boundaries visible to players
- **Engine integration**: Unreal Engine GDK, Unity GDK
- **Physics**: distributed simulation across hundreds of physics engines
- **Use case**: large persistent worlds (MMOs, simulation games)
- **Status**: pivoted from pure game platform to defense/simulation (Improbable Defence)
- **Criticism**: high complexity, vendor lock-in, some high-profile failures (Worlds Adrift shut down)

### EVE Online — Single-Shard Architecture (CCP Games)
- **Language**: Stackless Python (cooperative multitasking via tasklets and channels)
- **Tick rate**: 1 Hz (1 second per tick) — intentionally slow for MMO scale
- **Time Dilation (TiDi)**:
  - When a solar system's server node is overloaded, simulation slows to min 10% speed
  - At 10% TiDi: 1 second of game time = 10 seconds real time
  - All player actions slow proportionally — fair for everyone in the system
  - Preserves simulation integrity: no dropped inputs, no desyncs
- **Cluster**: ~30 standard nodes for the universe, 6 "reinforced" nodes for fleet fights and trade hubs
- **Record**: 24,000+ concurrent users on single shard (Tranquility)
- **Battles**: B-R5RB (2014) — 7,548 players in one system, TiDi at 10% for 21 hours
- **Architecture**: each solar system runs on one node; popular systems get dedicated reinforced hardware
- **Key insight**: TiDi is a brilliant design choice — instead of dropping packets or desyncing,
  the game literally slows down, preserving the simulation contract

---

## 7. Elo / Glicko / TrueSkill Matchmaking

### Elo Rating System

```
// Expected score (probability of winning):
E_a = 1 / (1 + 10^((R_b - R_a) / 400))

// Rating update after a game:
R_a_new = R_a + K * (S_a - E_a)
// S_a = 1 (win), 0.5 (draw), 0 (loss)

// K-factor (how much ratings change per game):
// FIDE chess: K=40 (new players), K=20 (established), K=10 (elite >2400)
// Games: typically K=32 for new players, K=16 for established
```

**Elo limitations for games:**
- Designed for 1v1 chess; doesn't handle teams or free-for-all
- No confidence measure — can't distinguish "played 5 games" from "played 5000"
- Rating inflation/deflation over time in player pools

### Glicko-2 Rating System (Mark Glickman)

```
// Three parameters per player:
//   μ (mu)    — rating (default: 1500, maps to ~25 in Glicko-2 internal scale)
//   φ (phi)   — rating deviation (confidence; default: 350, high = uncertain)
//   σ (sigma) — volatility (how erratic the player's performance is)

// Key innovation: rating deviation INCREASES over time if player is inactive
//   → inactive players have high φ, so their rating changes more when they return
//   → active players have low φ, so their rating is more stable

// Glicko-2 update steps (per rating period):
// 1. Convert to Glicko-2 scale: μ' = (r - 1500) / 173.7178
// 2. Compute variance from opponents' ratings
// 3. Compute performance delta
// 4. Update volatility σ via iterative algorithm (Illinois method)
// 5. Update φ* = sqrt(φ² + σ²)  (deviation grows with volatility)
// 6. Update φ and μ based on game results
// 7. Convert back to Glicko scale
```

**Used by**: Lichess, Team Fortress 2, Splatoon 2, CSGO (modified), Pokémon Showdown

### TrueSkill (Microsoft, Xbox)

```
// Two parameters per player:
//   μ (mu)    — mean skill (default: 25)
//   σ (sigma) — uncertainty (default: 25/3 ≈ 8.333)
//   Display rating = μ - 3σ (conservative estimate, starts at 0)

// Key innovations:
// - Handles teams and free-for-all (not just 1v1)
// - Bayesian inference via factor graphs + expectation propagation
// - Message passing between skill nodes, performance nodes, and outcome nodes
// - Draw probability is explicit (not just S=0.5)

// Matchmaking quality:
//   quality = exp(-Σ(μ_i - μ_j)² / (2 * Σ(2σ² + β²)))
//   Closer to 1.0 = better match

// TrueSkill 2 (2018):
// - Models individual performance across games (not just win/loss)
// - Accounts for game mode, map, time since last game
// - Uses Thurston model with Gaussian score distribution
```

**Used by**: Xbox Live, Halo, Gears of War

### Queue Management & SBMM Issues

**Queue management patterns:**
- **Expanding search**: start with tight skill range (±50), expand every 5 seconds
- **Backfill**: join in-progress games to replace disconnects; tag backfilled players for
  lenient rating changes
- **Priority queue**: premade groups weighted differently; solo players protected from 5-stacks
- **Ping bucketing**: first filter by region/ping, then by skill within acceptable latency

**SBMM controversy:**
- Players complain every game feels "sweaty" — no casual stomps
- "Engagement Optimized Matchmaking" (EOMM) conspiracy: some patents suggest matching
  players to maximize playtime/spending, not just skill
- Smurf detection: rapid skill convergence for new accounts performing above their bracket
- "Protected bracket" for new/low-skill players (first 10-50 games)

---

## 8. Anti-Cheat Approaches

### Server Authority — The Foundation

The #1 anti-cheat is not sending cheatable data in the first place:

```
// WRONG: client says "I have 999 health" → server accepts
// RIGHT: server tracks health; client sends inputs; server computes results

// WRONG: send all enemy positions to all clients
// RIGHT: only send positions of enemies the client can see
```

### Speed Hack Detection

```
// Server tracks player position each tick
// speed = distance(pos_new, pos_old) / dt
// If speed > max_allowed_speed * 1.1 (tolerance for network jitter):
//   → flag for review, don't teleport them, snap back to last valid position

// Also check distance over longer windows (1s, 5s) to catch gradual speed hacks
// that stay just under per-tick threshold

distance_1s = distance(pos_now, pos_1s_ago);
if distance_1s > MAX_SPEED * 1.0 * 1.15:  // 15% tolerance
    flag_player(SPEED_HACK)
```

### Teleport Detection

```
// Per-tick position delta check:
delta = distance(pos_new, pos_old);
if delta > MAX_SINGLE_TICK_MOVEMENT:  // e.g., 10 units at 60Hz
    if not player.has_teleport_ability():
        reject_movement()
        snap_to_last_valid()
```

### Wallhack Mitigation — Server-Side Visibility

**Valve PVS (Potentially Visible Set):**
- Pre-compiled visibility data baked into map BSP
- Server checks: "is entity in this player's PVS?"
- If not, don't include in snapshot → wallhack sees nothing
- Fast (lookup table) but conservative (may send entities behind nearby walls)

**Valorant Fog of War (state of the art):**
- PVS lookup + ray-cast verification
- Server withholds ALL data about enemies outside line of sight
- Audio footsteps: separate system — can trigger position reveal for nearby enemies
- Performance benefit: fewer entities to serialize = smaller packets
- Based on League of Legends' fog-of-war, adapted from 2D to 3D
- Uses voxelized PVS: divide world into cells, precompute cell-to-cell visibility
  - Lookup is O(1) — just check the table
  - Optimistic: if any part of voxel A can see any part of voxel B, mark visible
  - Avoids pop-in by being conservative at voxel boundaries

**Corner Culling (research):**
- Analytical ray casts against occluder geometry
- Can protect 50+ players with thousands of occluders in battle royale
- More precise than PVS but more computationally expensive

### Replay Analysis / Statistical Detection

- Record all player inputs server-side
- Offline analysis: aim-snap detection (inhuman crosshair acceleration), reaction time
  distributions, percentage of time crosshair is on enemy through walls
- Machine learning on behavioral data (aim patterns, movement, timing)
- Ban waves: collect data, ban in bulk to avoid revealing detection methods

### Kernel-Level Anti-Cheat

| System    | Developer  | Level       | Boot-Time | Games Using           |
|-----------|-----------|-------------|-----------|----------------------|
| Vanguard  | Riot      | Kernel (ring 0) | Yes (always on) | Valorant, League     |
| EAC       | Epic      | Kernel      | No (game launch) | Fortnite, Apex, Elden Ring |
| BattlEye  | BattlEye  | Kernel      | No (game launch) | PUBG, R6 Siege, DayZ |

**Vanguard specifics:**
- Runs at boot as a Windows service — monitors ALL driver loads after it
- Allowlist model: blocks unknown kernel drivers that could be cheat loaders
- Shadow memory: hides memory pages from other processes
- Uses `SwapContext()` hook to detect process manipulation
- Binary encryption (proprietary packer) resists static/dynamic analysis
- Controversial: always-on ring 0 access raises privacy/security concerns

**What they detect:**
- Process injection (DLL injection, manual mapping)
- Memory read/write from external processes
- Driver-based cheats (custom kernel drivers)
- Virtual machine introspection (running game in VM to read memory externally)
- Timing anomalies (speed hacks via clock manipulation)

---

## 9. Game Server Operations

### Metrics to Monitor

```yaml
# Essential game server metrics
tick_rate:
  target: 60  # or whatever your game needs
  alert: < 55  # degraded
  critical: < 40  # unplayable

tick_duration_ms:
  p50: 4-8     # healthy
  p99: < 16.67  # must fit in tick budget
  alert: p99 > 12  # approaching budget

player_count:
  per_server: gauge
  alert: > capacity * 0.9

entity_count:
  per_server: gauge
  alert: > 10000  # depends on game

network_latency_ms:
  p50: < 50
  p99: < 150
  alert: p99 > 200

bandwidth_out_mbps:
  per_server: gauge
  alert: > 80% of NIC capacity

memory_mb:
  per_server: gauge
  alert: > 80% of available

# Business metrics
matchmaking_wait_time_s:
  p50: < 30
  p99: < 120

player_disconnect_rate:
  per_minute: counter
  alert: > 5% of active players
```

**Histogram buckets for tick duration** (from OneUpTime blog):
- For a 60fps game (16.67ms budget): `[1, 2, 4, 8, 12, 16, 20, 33, 50, 100]` ms
- Most ticks should land in 4-16ms buckets
- Ticks hitting 33ms or 50ms indicate a serious problem
- Use OpenTelemetry custom metrics for cross-platform observability

### Hot Reload for Game Logic

**Approaches:**
1. **Scripting layer** (Lua, TypeScript via V8/QuickJS):
   - Reload scripts without restarting server process
   - Nakama uses Go/TS/Lua; can hot-reload Lua scripts
   - EVE Online: Stackless Python tasklets can be updated via code push
   - Overhead: script interpreter adds ~10-30% CPU vs native

2. **Shared library swap** (C/C++/Rust):
   - Compile game logic as `.so`/`.dll`, `dlopen`/`dlclose` at runtime
   - Must serialize all state, unload old lib, load new lib, deserialize
   - Risk: ABI breakage if struct layouts change
   - Used by: some Unreal Engine servers, custom engines

3. **Process migration** (GoWorld pattern):
   - `SIGHUP` to old process → serializes all state to shared memory/file
   - Start new process with `--restore` flag → deserializes state
   - Transparent to connected players (connection handoff via SO_REUSEPORT or fd passing)
   - GoWorld (Go): full hot-swap with player connection preservation

4. **Rolling restart with matchmaker drain**:
   - Most practical: tell matchmaker to stop sending new players to server X
   - Wait for current match to end (or migrate players)
   - Restart with new binary

### Graceful Shutdown

```
// Signal handling:
on SIGTERM:
  1. Stop accepting new connections (tell matchmaker: "I'm draining")
  2. Notify connected players: "Server shutting down in 60s"
  3. Save all player state to database
  4. Wait for in-flight RPCs to complete (timeout: 30s)
  5. Close all connections
  6. Flush metrics/logs
  7. Exit 0

// Colyseus pattern:
on SIGTERM/SIGINT:
  1. gameServer.onBeforeShutdown()
  2. All rooms: room.onBeforeShutdown()  // may keep room alive for a few minutes
  3. room.disconnect() for all clients
  4. room.onLeave() per client
  5. room.onDispose()
  6. Process exit
```

### Blue-Green Deployment for Game Servers

```
                    ┌─────────────┐
                    │ Matchmaker / │
                    │ Load Balancer│
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │                         │
        ┌─────┴─────┐           ┌──────┴─────┐
        │  BLUE      │           │  GREEN     │
        │  (v1.2)    │           │  (v1.3)    │
        │  ACTIVE    │           │  STANDBY   │
        └────────────┘           └────────────┘

// Deployment steps:
1. Deploy v1.3 to GREEN fleet (servers are idle)
2. Smoke test GREEN: synthetic players, integration tests
3. Flip matchmaker: new matches go to GREEN
4. BLUE drains: existing matches play out (no new matches)
5. When BLUE is empty: decommission or hold for rollback
6. If GREEN has issues: flip matchmaker back to BLUE (<1 min rollback)
```

**Game-specific considerations:**
- Can't cut active game sessions — must drain gracefully
- Match length matters: a 5-min FPS match drains fast; an MMO raid can take hours
- Persistent world: use database for state transfer, not in-memory
- Regional deployment: roll out one region at a time (US-West → US-East → EU → Asia)

---

## 10. Serialization for Games

### FlatBuffers (Google, zero-copy)

```
// Schema:
table Monster {
  pos: Vec3;
  hp: short = 100;
  name: string;
  inventory: [ubyte];
}

// Zero-copy read — no deserialization step:
let monster = root_as_monster(buffer);
let hp = monster.hp();  // direct pointer arithmetic into buffer
// No allocations, no copies, no parsing
```

- **Access time**: O(1) random access to any field via vtable offset
- **Write**: must build bottom-up (strings/vectors first, then tables)
- **Size**: slightly larger than protobuf (no variable-length int encoding)
- **Used by**: Cocos2d-x (all game data), Android (IPC), Facebook (Messenger)
- **Game advantage**: server can write once, send same bytes to all clients

### Cap'n Proto

- Wire format IS the in-memory format — truly zero-copy in both directions
- Pointer-based navigation (vs FlatBuffers' vtable-based)
- Built-in RPC framework
- Arena-style allocation required (contiguous memory for message building)
- No optional fields (use unions instead) — more rigid but faster
- Better for: internal server-to-server communication

### Custom Binary Protocols with Bit-Packing

```rust
// Bit writer (Gaffer on Games pattern):
struct BitWriter {
    buffer: Vec<u32>,
    scratch: u64,      // double-width scratch space
    scratch_bits: u32,  // bits used in scratch
}

impl BitWriter {
    fn write_bits(&mut self, value: u32, bits: u32) {
        self.scratch |= (value as u64) << self.scratch_bits;
        self.scratch_bits += bits;
        if self.scratch_bits >= 32 {
            self.buffer.push(self.scratch as u32);
            self.scratch >>= 32;
            self.scratch_bits -= 32;
        }
    }

    // Domain-specific helpers:
    fn write_bool(&mut self, v: bool)   { self.write_bits(v as u32, 1); }
    fn write_health(&mut self, hp: u16) { self.write_bits(hp as u32, 10); } // 0-1023
    fn write_angle(&mut self, deg: f32) { // 0-360 in 10 bits = 0.35° precision
        let quantized = ((deg / 360.0) * 1024.0) as u32;
        self.write_bits(quantized & 0x3FF, 10);
    }
    fn write_pos_delta(&mut self, dx: i16) { // variable-length small delta
        if dx == 0 { self.write_bits(0, 1); return; }
        self.write_bits(1, 1);
        if dx.abs() < 16 {
            self.write_bits(0, 1);
            self.write_bits((dx + 16) as u32, 5); // 5 bits: -16..+15
        } else {
            self.write_bits(1, 1);
            self.write_bits(dx as u32 & 0xFFFF, 16); // full 16-bit
        }
    }
}
```

### Bandwidth Savings Example

```
// Player state (naive encoding): 48 bytes
struct PlayerState {
    position: [f32; 3],   // 12 bytes
    velocity: [f32; 3],   // 12 bytes
    yaw: f32,             // 4 bytes
    pitch: f32,           // 4 bytes
    health: u16,          // 2 bytes
    armor: u16,           // 2 bytes
    weapon_id: u8,        // 1 byte
    flags: u8,            // 1 byte (crouching, jumping, firing, etc.)
    ammo: u16,            // 2 bytes
    // + padding          // ~8 bytes
}                         // Total: ~48 bytes

// Bit-packed (with delta compression against last acked state):
// changed_mask:     8 bits   (which fields changed)
// position delta:   3 × 12 bits = 36 bits (±2048 units, 0.5 unit precision)
// velocity:         usually skip (predicted from position deltas)
// yaw delta:        10 bits (0.35° precision)
// pitch delta:      9 bits  (-90..+90°, 0.35° precision)
// health:           10 bits (0-1023)
// armor:            7 bits  (0-100)
// weapon_id:        4 bits  (16 weapons)
// flags:            6 bits
// ammo:             8 bits  (0-255)
// Total changed:    ~98 bits = 13 bytes
// With most fields unchanged: typically 2-5 bytes per entity per tick

// 64 players × 5 bytes × 60 Hz = 19.2 KB/s vs naive 64 × 48 × 60 = 184 KB/s
// That's ~10x compression
```

### Variable-Length Encoding Patterns

```
// Varint (protobuf-style): 7 bits per byte, MSB = "more bytes follow"
fn write_varint(value: u64) -> Vec<u8> {
    let mut bytes = vec![];
    let mut v = value;
    while v >= 0x80 {
        bytes.push((v as u8 & 0x7F) | 0x80);
        v >>= 7;
    }
    bytes.push(v as u8);
    bytes
}
// 0-127: 1 byte, 128-16383: 2 bytes, etc.

// Small-value optimization: most game values are small
//   Entity IDs: usually < 1024 (10 bits)
//   Player count: < 128 (7 bits)
//   Map coordinates: quantize to grid, send grid cell index
```

### Oodle Network (RAD Game Tools)

Production games use Oodle Network Compression on top of custom protocols:
- Learns patterns in your game's packet data (trains a dictionary)
- 2-3x compression on top of delta+bitpacking
- Used by Fortnite, many AAA titles
- Compresses individual UDP packets (not stream-based)

---

## Quick Reference: Architecture Decision Matrix

| Factor | Room-Based (Photon) | Relay (Nakama) | Distributed (SpatialOS) | Single-Shard (EVE) |
|--------|---------------------|----------------|------------------------|-------------------|
| Players/instance | 2-64 | 2-100 | 1000s | 1000s |
| World persistence | No | Optional | Yes | Yes |
| Server authority | Optional | Optional | Yes | Yes |
| Complexity | Low | Medium | Very High | Very High |
| Best for | Lobby games | Social/mobile | Open-world MMO | Persistent MMO |
| Hosting | Cloud | Self/Cloud | Cloud only | Self-hosted |

---

## Sources

- [Gaffer On Games — Fix Your Timestep!](https://gafferongames.com/post/fix_your_timestep/)
- [Gaffer On Games — Reading and Writing Packets](https://gafferongames.com/post/reading_and_writing_packets/)
- [Gaffer On Games — Snapshot Compression](https://gafferongames.com/post/snapshot_compression/)
- [Gabriel Gambetta — Client-Side Prediction](https://www.gabrielgambetta.com/client-side-prediction-server-reconciliation.html)
- [Gabriel Gambetta — Entity Interpolation](https://www.gabrielgambetta.com/entity-interpolation.html)
- [Valve Developer Wiki — Source Multiplayer Networking](https://developer.valvesoftware.com/wiki/Source_Multiplayer_Networking)
- [Valve Developer Wiki — Lag Compensation](https://developer.valvesoftware.com/wiki/Lag_Compensation)
- [GDC Vault — Overwatch Gameplay Architecture and Netcode](https://gdcvault.com/play/1024001/-Overwatch-Gameplay-Architecture-and)
- [GDC Vault — I Shot You First: Halo Reach](https://www.gdcvault.com/play/1014345/I-Shot-You-First-Networking)
- [Riot Games — Demolishing Wallhacks with Fog of War](https://technology.riotgames.com/news/demolishing-wallhacks-valorants-fog-war)
- [Quake 3 Source Code Review — Network Model](https://fabiensanglard.net/quake3/network.php)
- [Quake 3 Networking Model (bookofhook)](https://fabiensanglard.net/quake3/The%20Quake3%20Networking%20Mode.html)
- [KCP Protocol — GitHub](https://github.com/skywind3000/kcp/blob/master/README.en.md)
- [Bevy ECS — Tainted Coders](https://taintedcoders.com/bevy/ecs)
- [Bevy ECS — DeepWiki](https://deepwiki.com/bevyengine/bevy/2-entity-component-system-(ecs))
- [SanderMertens — ECS FAQ](https://github.com/SanderMertens/ecs-faq)
- [EVE Online — Introducing Time Dilation](https://www.eveonline.com/news/view/introducing-time-dilation-tidi)
- [EVE Online — Time Dilation: How's That Going?](https://www.eveonline.com/news/view/time-dilation-hows-that-going)
- [Stackless Python in EVE — SlideShare](https://www.slideshare.net/Arbow/stackless-python-in-eve)
- [TrueSkill — Microsoft Research](https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/)
- [TrueSkill 2 — Microsoft Research](https://www.microsoft.com/en-us/research/publication/trueskill-2-improved-bayesian-skill-rating-system/)
- [Glicko-2 Example — Professor Mark Glickman](https://www.glicko.net/glicko/glicko2.pdf)
- [Glicko-2 for Game Ratings — GitHub Gist](https://gist.github.com/gpluscb/302d6b71a8d0fe9f4350d45bc828f802)
- [Photon Server FAQ](https://doc.photonengine.com/server/current/reference/faq)
- [Nakama — Heroic Labs](https://heroiclabs.com/nakama/)
- [Mirror Networking](https://mirror-networking.com/)
- [SpatialOS — Improbable](https://ims.improbable.io/products/spatialos/)
- [FlatBuffers — Wikipedia](https://en.wikipedia.org/wiki/FlatBuffers)
- [Cap'n Proto vs FlatBuffers](https://capnproto.org/news/2014-06-17-capnproto-flatbuffers-sbe.html)
- [Anti-Cheat Examination — ACM](https://dl.acm.org/doi/fullHtml/10.1145/3664476.3670433)
- [SigNoz — Game Server Monitoring](https://signoz.io/guides/game-server-monitoring/)
- [OneUpTime — Monitor Game Server Tick Rate with OpenTelemetry](https://oneuptime.com/blog/post/2026-02-06-monitor-game-server-tick-rate-opentelemetry/view)
- [GoWorld — Hot Swapping Game Server](https://github.com/xiaonanln/goworld)
- [Colyseus — Graceful Shutdown](https://docs.colyseus.io/server/graceful-shutdown)
- [CS2 Tick Rate & Subtick Explained](https://blix.gg/news/cs-2/cs2-tick-rate-subtick-explained-64-hz-vs-128-tick-faceit-update-2025/)
- [Daposto — Game Networking: Compression & Bit Packing](https://daposto.medium.com/game-networking-5-compression-delta-encoding-interest-management-bit-packing-9316ff1c96db)
- [Hot Reloading in Game Server Development](https://www.oreateai.com/blog/hot-reloading-technology-in-game-server-development-part-1-basic-principles-and-application-scenarios/37c0e7727beb0e4a538e0bace4c51327)
