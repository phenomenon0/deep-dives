# Cryptography Deep Dive — Research Notes

Compiled 2026-03-15. Structured for direct use in 14-chapter HTML content.

---

## 1. Signal Protocol

### X3DH (Extended Triple Diffie-Hellman) Key Agreement

**Curves:** All public keys must be either X25519 or X448 (same curve for entire run). Signal uses X25519 in practice.

**Key types per user:**
- **Identity Key (IK):** Long-term Curve25519 key pair, generated once at install
- **Signed Prekey (SPK):** Medium-term key, rotated periodically (typically every 1-7 days), signed by IK
- **One-Time Prekeys (OPK):** Ephemeral, each used exactly once then deleted. **100 uploaded at install** (`KeyHelper.generatePreKeys(startId, 100)`), replenished when server signals supply is low
- **Ephemeral Key (EK):** Generated fresh by sender for each new session

**Prekey bundle (published to server by Bob):**
- Bob's identity key IK_B
- Bob's signed prekey SPK_B + signature
- One of Bob's one-time prekeys OPK_B (consumed and deleted after use)

**DH calculations (Alice initiating to Bob):**
```
DH1 = DH(IK_A, SPK_B)    — Alice's identity ↔ Bob's signed prekey
DH2 = DH(EK_A, IK_B)     — Alice's ephemeral ↔ Bob's identity
DH3 = DH(EK_A, SPK_B)    — Alice's ephemeral ↔ Bob's signed prekey
DH4 = DH(EK_A, OPK_B)    — Alice's ephemeral ↔ Bob's one-time prekey (if available)

SK = HKDF(DH1 || DH2 || DH3 || DH4)   — 32-byte shared secret
```
If no OPK_B available, DH4 is omitted: `SK = HKDF(DH1 || DH2 || DH3)`

**Security roles of each DH:**
- DH1 + DH2: Mutual authentication (both identity keys involved)
- DH3: Forward secrecy (ephemeral ↔ medium-term)
- DH4: Additional forward secrecy + one-time prekey consumption prevents replay

**Cryptographic primitives:** Curve25519, AES-256, HMAC-SHA256

### Double Ratchet Algorithm

**Three KDF chains:**
1. **Root chain** — produces new chain keys for sending/receiving chains
2. **Sending chain** — derives per-message encryption keys
3. **Receiving chain** — derives per-message decryption keys

**Symmetric ratchet step (within one chain):**
```
(new_chain_key, message_key) = HKDF(chain_key, constant)
```
Every message gets a unique message key. Chain key advances forward — old message keys cannot be derived from new chain key (forward secrecy within a chain).

**DH ratchet step (Diffie-Hellman ratchet):**
- Triggered when a new ratchet public key arrives from the other party
- Each party generates a new ephemeral DH key pair
- New DH shared secret feeds into root chain KDF
- Root chain outputs become new sending/receiving chain keys
- "Ping-pong" pattern: parties alternate generating new ratchet keys

**KDF chain properties (three guarantees):**
1. **Resilience:** Output keys appear random without knowing KDF keys
2. **Forward security:** Past output keys appear random even if current KDF key is compromised
3. **Break-in recovery:** Future output keys appear random after compromise, once new entropy enters

**Forward secrecy guarantee:** Compromising current state does NOT reveal past message keys (already deleted from chain).

**Post-compromise security timeline:**
- Recovery requires **one DH ratchet step** (ΔCK = 1)
- In practice: attacker compromises Alice → Alice sends a message → Bob replies (new DH ratchet key) → Alice sends again (new DH ratchet key) → compromise healed
- Concretely: **2-3 messages exchanged** between parties restores security
- The compromised ratchet private key is replaced with an uncompromised one through the ping-pong DH ratchet

### PQXDH (Post-Quantum Extension)
- Signal added ML-KEM (Kyber) to X3DH in 2023
- Adds a 5th DH-equivalent using KEM encapsulation
- Post-quantum forward secrecy, but more messages needed for post-compromise security recovery

---

## 2. TLS 1.3 Handshake

### Full 1-RTT Handshake Message Flow

```
Client                                           Server

ClientHello
  + key_share
  + supported_versions
  + signature_algorithms          -------->
                                                  ServerHello
                                                    + key_share
                                                    + supported_versions
                                              {EncryptedExtensions}
                                              {CertificateRequest*}
                                              {Certificate}
                                              {CertificateVerify}
                                              {Finished}
                                  <--------
{Certificate*}
{CertificateVerify*}
{Finished}                        -------->
[Application Data]                <------->   [Application Data]
```

`{}` = encrypted with handshake keys, `[]` = encrypted with application keys, `*` = optional

**Key insight:** Client sends DH key_share in ClientHello (before knowing server's choice). Server responds with its key_share in ServerHello. Shared secret is derived immediately. Everything after ServerHello is encrypted.

### What Changed from TLS 1.2

| Feature | TLS 1.2 | TLS 1.3 |
|---------|---------|---------|
| Round trips | 2-RTT | 1-RTT (0-RTT for resumption) |
| Key exchange | RSA static, DHE, ECDHE | **ECDHE only** (forward secrecy mandatory) |
| Cipher suites | 37+ negotiable | 5 AEAD-only suites |
| RSA key transport | Supported | **Removed entirely** |
| CBC mode ciphers | Supported | **Removed** (POODLE, Lucky13) |
| RC4, 3DES | Supported | **Removed** |
| MD5, SHA-1 in PRF | Supported | **Removed** |
| Compression | Supported | **Removed** (CRIME attack) |
| Renegotiation | Supported | **Removed** |
| Session tickets | Opaque | PSK-based resumption |
| Cipher suite format | `TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256` | `TLS_AES_128_GCM_SHA256` (no key exchange in name) |

**Why 1-RTT matters:**
- TLS 1.2: ClientHello → ServerHello+Certificate → ClientKeyExchange → ChangeCipherSpec+Finished = 2 round trips before data
- TLS 1.3: Client sends key_share speculatively in first message. If server accepts, handshake completes in 1 RTT
- At 100ms RTT, saves 100ms per new connection. At scale (millions of connections), enormous latency reduction

### 0-RTT (Early Data) — Risks

**How it works:** Client with a PSK (pre-shared key) from a previous session sends encrypted application data alongside ClientHello, before the handshake completes.

**The critical risk — replay attacks:**
- 0-RTT data is **NOT forward-secret** (encrypted under PSK, not fresh DH)
- **No replay protection** between connections — attacker can record and replay the ClientHello + early data
- The early data often contains the most sensitive payload: cookies, auth tokens, credit card numbers
- Server has no cryptographic guarantee the early data is fresh

**Mitigations:**
- Only allow 0-RTT for **idempotent operations** (e.g., GET requests, not POST)
- Server-side replay detection via nonce/timestamp tracking
- Cloudflare, nginx can be configured to reject 0-RTT entirely or limit it
- RFC 8446 Section 8: "any server which receives 0-RTT data MUST have a strategy for handling replays"

---

## 3. Real-World CVEs

### Heartbleed — CVE-2014-0160

**The bug:** Missing bounds check on `memcpy()` in OpenSSL's TLS Heartbeat extension handler.

**Exact mechanism:**
```c
/* Attacker sends: heartbeat request with payload "hello" (5 bytes)
   but claims payload_length = 65535 (max) */

/* Vulnerable code (simplified): */
memcpy(bp, pl, payload);
/* 'payload' comes from attacker-controlled length field
   No check: actual_data_length >= claimed_payload_length
   Server copies 64KB of adjacent process memory into response */
```

**Impact per request:** Up to **64KB of server memory** leaked — private keys, session cookies, passwords, other users' data

**Timeline:**
- **2012-03-14:** Vulnerable code introduced in OpenSSL 1.0.1 (commit by Robin Seggelmann)
- **2014-03-21:** Neel Mehta (Google Security) discovers the bug
- **2014-04-01:** Codenomicon independently discovers it
- **2014-04-07:** Public disclosure + OpenSSL 1.0.1g released (same day)
- **~2 years in the wild** undetected

**Scope:** Estimated **17% of all SSL web servers** (those using OpenSSL 1.0.1–1.0.1f) affected. ~500,000 servers.

**War story:** The Canadian Revenue Agency confirmed attackers used Heartbleed to extract 900 SINs (Social Insurance Numbers) before the patch. Cloudflare initially claimed private key extraction was impossible, then had to retract when researchers demonstrated it within hours of the "Heartbleed Challenge."

### POODLE — CVE-2014-3566

**Full name:** Padding Oracle On Downgraded Legacy Encryption

**The bug:** SSLv3's CBC mode padding is **not covered by the MAC** (Message Authentication Code). The last block's padding bytes are undefined except for the final byte (which must equal the padding length).

**Attack mechanism:**
1. Attacker forces TLS downgrade to SSLv3 (via connection interference)
2. Manipulates ciphertext blocks to move a target byte (e.g., cookie character) to the padding position
3. If server accepts the modified record (1/256 chance per byte), the padding byte reveals the plaintext byte
4. **256 requests per byte** on average to decrypt one byte
5. To steal a 16-byte session cookie: ~4,096 requests

**Timeline:**
- **2014-10-14:** Disclosed by Google researchers (Bodo Möller, Thai Duong, Krzysztof Kotowicz)
- SSLv3 was already 18 years old at disclosure

**Legacy impact:** Effectively killed SSLv3. Led to RFC 7568 ("Deprecating SSLv3").

### Goto Fail — CVE-2014-1266

**The bug:** A duplicate `goto fail;` statement in Apple's Secure Transport library.

**Exact code (from `sslKeyExchange.c`, function `SSLVerifySignedServerKeyExchange`):**
```c
if ((err = SSLHashSHA1.update(&hashCtx, &serverRandom)) != 0)
    goto fail;
if ((err = SSLHashSHA1.update(&hashCtx, &signedParams)) != 0)
    goto fail;
    goto fail;  /* ← DUPLICATE: always executes, skips signature verification */
if ((err = SSLHashSHA1.final(&hashCtx, &hashOut)) != 0)
    goto fail;
```

**Why it's devastating:** The second `goto fail` is not inside any `if` block (no curly braces). It executes unconditionally. At the `fail:` label, `err` still holds 0 (success from the last `update` call). The function returns success **without ever verifying the server's signature**.

**Impact:** Any MITM attacker could present a fraudulent certificate and Apple devices (iOS 7.0.x, OS X 10.9.x) would accept it. Complete TLS MITM on all affected Apple devices.

**Timeline:**
- **2014-02-21:** Apple silently patches iOS 7.0.6
- **2014-02-22:** Security researchers notice and sound alarm — OS X still unpatched
- **2014-02-25:** OS X 10.9.2 patch released
- Root cause likely: merge conflict or bad cherry-pick introducing duplicate line

**Lessons:** No curly braces on if-statements, missing code coverage (the code after the duplicate goto was dead code), no compiler warnings for unreachable code in that build config.

### ROCA — CVE-2017-15361

**Full name:** Return of Coppersmith's Attack

**The bug:** Infineon's RSALib (in TPMs, smartcards, YubiKey 4) used a "Fast Prime" generation algorithm that produced RSA primes with a specific mathematical structure, making them factorable.

**Key facts:**
- Vulnerability does NOT depend on weak RNG — **all keys from vulnerable chips are affected**
- Only the **public key** is needed to recover the private key (no physical access required)
- Present in **FIPS 140-2 and CC EAL 5+ certified devices** since at least 2012
- Affected: Infineon TPMs, YubiKey 4 (RSA keys generated on-chip), Estonian national ID cards

**Factoring costs (Intel E5-2650 v3@3GHz, 2014 pricing):**
| Key size | CPU time | Cost (AWS) |
|----------|----------|------------|
| 512-bit | 2 CPU hours | $0.06 |
| 1024-bit | 97 CPU days | $40–$80 |
| 2048-bit | 140.8 CPU years | $20,000–$40,000 |

**Timeline:**
- **2017-10-16:** Public disclosure by CRoCS lab (Masaryk University)
- **2017-10-30:** Full paper presented at ACM CCS '17
- Estonia temporarily suspended 750,000 national ID cards

---

## 4. Library APIs — 5 Languages

### C: libsodium

```c
#include <sodium.h>

/* Initialize (required once) */
if (sodium_init() < 0) { /* panic */ }

/* === Symmetric authenticated encryption (XSalsa20-Poly1305) === */
unsigned char key[crypto_secretbox_KEYBYTES];       /* 32 bytes */
unsigned char nonce[crypto_secretbox_NONCEBYTES];    /* 24 bytes */
crypto_secretbox_keygen(key);
randombytes_buf(nonce, sizeof nonce);
/* Encrypt: ciphertext = 16-byte MAC + encrypted message */
crypto_secretbox_easy(ciphertext, message, message_len, nonce, key);
/* Decrypt: returns -1 if MAC verification fails */
if (crypto_secretbox_open_easy(decrypted, ciphertext, ciphertext_len, nonce, key) != 0) {
    /* FORGERY DETECTED */
}

/* === Public-key authenticated encryption (X25519 + XSalsa20-Poly1305) === */
unsigned char alice_pk[crypto_box_PUBLICKEYBYTES];   /* 32 bytes */
unsigned char alice_sk[crypto_box_SECRETKEYBYTES];   /* 32 bytes */
crypto_box_keypair(alice_pk, alice_sk);
/* crypto_box_easy(c, m, mlen, nonce, bob_pk, alice_sk) */
/* crypto_box_open_easy(m, c, clen, nonce, alice_pk, bob_sk) */

/* === Signatures (Ed25519) === */
unsigned char sign_pk[crypto_sign_PUBLICKEYBYTES];   /* 32 bytes */
unsigned char sign_sk[crypto_sign_SECRETKEYBYTES];   /* 64 bytes */
crypto_sign_keypair(sign_pk, sign_sk);
/* crypto_sign_detached(sig, &siglen, m, mlen, sk) — 64-byte signature */
/* crypto_sign_verify_detached(sig, m, mlen, pk)   — returns 0 if valid */

/* === Password hashing (Argon2id under the hood) === */
char hash[crypto_pwhash_STRBYTES];  /* 128 bytes, includes algo+params+salt+hash */
crypto_pwhash_str(hash, password, strlen(password),
    crypto_pwhash_OPSLIMIT_MODERATE,    /* 3 iterations */
    crypto_pwhash_MEMLIMIT_MODERATE);   /* 256 MB */
/* Verify: */
if (crypto_pwhash_str_verify(hash, password, strlen(password)) != 0) {
    /* WRONG PASSWORD */
}
```

**Key constants:**
- `crypto_secretbox_KEYBYTES` = 32, `_NONCEBYTES` = 24, `_MACBYTES` = 16
- `crypto_box_PUBLICKEYBYTES` = 32, `_SECRETKEYBYTES` = 32
- `crypto_sign_BYTES` = 64 (signature size)
- `crypto_pwhash_OPSLIMIT_INTERACTIVE` = 2, `_MODERATE` = 3, `_SENSITIVE` = 4
- `crypto_pwhash_MEMLIMIT_INTERACTIVE` = 64MB, `_MODERATE` = 256MB, `_SENSITIVE` = 1GB

### Python: `cryptography` library

```python
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import os, hashlib

# === Fernet (AES-128-CBC + HMAC-SHA256, with timestamp) ===
key = Fernet.generate_key()         # 32-byte URL-safe base64
f = Fernet(key)
token = f.encrypt(b"secret data")   # Returns Fernet token (version|timestamp|IV|ciphertext|HMAC)
plaintext = f.decrypt(token)        # Raises InvalidToken on failure
# With TTL: f.decrypt(token, ttl=60) — rejects tokens older than 60 seconds

# === AES-GCM (from hazmat layer) ===
key = AESGCM.generate_key(bit_length=256)  # 32 bytes
aesgcm = AESGCM(key)
nonce = os.urandom(12)                      # 96-bit nonce (standard for GCM)
ct = aesgcm.encrypt(nonce, plaintext, associated_data)  # associated_data can be None
pt = aesgcm.decrypt(nonce, ct, associated_data)

# === Ed25519 Signatures ===
private_key = Ed25519PrivateKey.generate()
public_key = private_key.public_key()
signature = private_key.sign(b"message")    # 64-byte Ed25519 signature
public_key.verify(signature, b"message")    # Raises InvalidSignature on failure

# === hashlib (NOT for passwords — for integrity) ===
h = hashlib.sha256(b"data").hexdigest()
h = hashlib.blake2b(b"data", digest_size=32).hexdigest()
```

### TypeScript/Node: Web Crypto API + `node:crypto`

```typescript
// === Web Crypto API (works in browsers AND Node.js) ===
const { subtle } = globalThis.crypto;

// AES-GCM key generation
const key = await subtle.generateKey(
  { name: "AES-GCM", length: 256 },
  true,                              // extractable
  ["encrypt", "decrypt"]
);

// Encrypt
const iv = crypto.getRandomValues(new Uint8Array(12));  // 96-bit IV
const ciphertext = await subtle.encrypt(
  { name: "AES-GCM", iv },
  key,
  new TextEncoder().encode("secret message")
);

// Decrypt
const plaintext = await subtle.decrypt(
  { name: "AES-GCM", iv },
  key,
  ciphertext
);

// SHA-256 digest
const hash = await subtle.digest("SHA-256", new TextEncoder().encode("data"));
// Returns ArrayBuffer — convert: Buffer.from(hash).toString('hex')

// ECDH key agreement
const keyPair = await subtle.generateKey(
  { name: "ECDH", namedCurve: "P-256" },
  false,
  ["deriveKey"]
);

// Ed25519 (Node.js 18+)
// NOTE: Ed25519 support in Web Crypto is still evolving
const edKey = await subtle.generateKey("Ed25519", true, ["sign", "verify"]);

// === node:crypto (lower-level, Node.js-specific) ===
import { createCipheriv, createDecipheriv, randomBytes, createHash,
         scryptSync, createHmac } from "node:crypto";

const nodeKey = randomBytes(32);
const nodeIv = randomBytes(16);
const cipher = createCipheriv("aes-256-gcm", nodeKey, nodeIv);
// scryptSync(password, salt, keylen) for password-based key derivation
```

### Go: `crypto/*` + `golang.org/x/crypto`

```go
import (
    "crypto/aes"
    "crypto/cipher"
    "crypto/ed25519"
    "crypto/rand"
    "crypto/sha256"
    "golang.org/x/crypto/nacl/secretbox"
    "golang.org/x/crypto/nacl/box"
    "golang.org/x/crypto/argon2"
    "io"
)

// === AES-256-GCM ===
key := make([]byte, 32)
io.ReadFull(rand.Reader, key)
block, _ := aes.NewCipher(key)
gcm, _ := cipher.NewGCM(block)
nonce := make([]byte, gcm.NonceSize())  // 12 bytes
io.ReadFull(rand.Reader, nonce)
ciphertext := gcm.Seal(nonce, nonce, plaintext, nil)  // nonce prepended
// Decrypt: gcm.Open(nil, nonce, ciphertext, nil)

// === NaCl secretbox (XSalsa20-Poly1305) ===
var secretKey [32]byte
io.ReadFull(rand.Reader, secretKey[:])
var nonce24 [24]byte
io.ReadFull(rand.Reader, nonce24[:])
encrypted := secretbox.Seal(nonce24[:], message, &nonce24, &secretKey)
// Decrypt: secretbox.Open(nil, encrypted[24:], &nonce24, &secretKey)

// === NaCl box (X25519 + XSalsa20-Poly1305) ===
pubKey, privKey, _ := box.GenerateKey(rand.Reader)
// box.Seal(out, message, &nonce, peerPub, myPriv)
// box.Open(out, ciphertext, &nonce, peerPub, myPriv)

// === Ed25519 ===
pub, priv, _ := ed25519.GenerateKey(rand.Reader)
sig := ed25519.Sign(priv, message)       // 64-byte signature
ok := ed25519.Verify(pub, message, sig)  // returns bool

// === SHA-256 ===
hash := sha256.Sum256(data)  // [32]byte

// === Argon2id ===
salt := make([]byte, 16)
io.ReadFull(rand.Reader, salt)
dk := argon2.IDKey([]byte(password), salt, 1, 64*1024, 4, 32)
// params: time=1, memory=64MB, threads=4, keyLen=32
```

### Rust: `ring` + RustCrypto ecosystem

```rust
// === ring crate (Google's BoringSSL-derived, no_std-friendly) ===
use ring::{aead, digest, rand, signature};

// AES-256-GCM with ring
let rng = rand::SystemRandom::new();
let key_bytes = aead::UnboundKey::generate(&aead::AES_256_GCM, &rng)?;
let key = aead::LessSafeKey::new(key_bytes);
let nonce = aead::Nonce::try_assume_unique_for_key(&nonce_bytes)?;
key.seal_in_place_append_tag(nonce, aead::Aad::empty(), &mut in_out)?;
// Decrypt: key.open_in_place(nonce, aead::Aad::empty(), &mut in_out)?;

// SHA-256 with ring
let hash = digest::digest(&digest::SHA256, b"data");

// Ed25519 with ring
let pkcs8 = signature::Ed25519KeyPair::generate_pkcs8(&rng)?;
let key_pair = signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref())?;
let sig = key_pair.sign(b"message");
// Verify:
let peer_pub = signature::UnparsedPublicKey::new(&signature::ED25519, pub_bytes);
peer_pub.verify(b"message", sig.as_ref())?;

// === RustCrypto crates (pure Rust, trait-based) ===
use aes_gcm::{Aes256Gcm, KeyInit, aead::Aead};
use sha2::{Sha256, Digest};
use ed25519_dalek::{SigningKey, Signer, Verifier};

// AES-256-GCM
let key = Aes256Gcm::generate_key(&mut OsRng);
let cipher = Aes256Gcm::new(&key);
let ciphertext = cipher.encrypt(&nonce, plaintext.as_ref())?;
let plaintext = cipher.decrypt(&nonce, ciphertext.as_ref())?;

// SHA-256
let mut hasher = Sha256::new();
hasher.update(b"data");
let hash = hasher.finalize();  // GenericArray<u8, U32>

// Ed25519
let signing_key = SigningKey::generate(&mut OsRng);
let signature = signing_key.sign(b"message");
let verifying_key = signing_key.verifying_key();
verifying_key.verify(b"message", &signature)?;
```

**ring vs RustCrypto trade-offs:**
- `ring`: C/asm under the hood (BoringSSL), faster, audited, but limited algorithm set, not pure Rust
- RustCrypto: Pure Rust, broader algorithm coverage, trait-based composability, `no_std` support

---

## 5. AES Internals

### The Four Operations (each round)

**State:** 4×4 matrix of bytes (128 bits)

**1. SubBytes (non-linear substitution):**
- Each byte → multiplicative inverse in GF(2^8) → affine transformation
- Implemented as a 256-byte lookup table (the S-box)
- Provides **confusion** (complex relationship between key and ciphertext)
- The ONLY non-linear step — all AES security against algebraic attacks rests here

**2. ShiftRows (transposition):**
- Row 0: no shift
- Row 1: shift left 1 byte
- Row 2: shift left 2 bytes
- Row 3: shift left 3 bytes
- Ensures bytes from one column spread across all four columns over multiple rounds
- Provides **diffusion** across columns

**3. MixColumns (linear mixing):**
- Each column treated as a polynomial over GF(2^8)
- Multiplied by a fixed polynomial: `c(x) = 3x³ + x² + x + 2`
- Each output byte depends on all 4 input bytes of that column
- Provides **diffusion** within columns
- **Skipped in the final round** (round 10/12/14)

**4. AddRoundKey (XOR with round key):**
- Simple bitwise XOR of state with the round key (derived from key schedule)
- Only step that introduces key material

### Round counts:
- **AES-128:** 10 rounds (128-bit key, 44 32-bit words in expanded key)
- **AES-192:** 12 rounds (192-bit key)
- **AES-256:** 14 rounds (256-bit key)

### Key Schedule:
- Original key expanded into `4 × (Nr + 1)` 32-bit words
- AES-128: 16 bytes → 176 bytes (11 round keys)
- Uses RotWord, SubWord (reusing S-box), XOR with round constant (Rcon)

### The ECB Penguin

The famous demonstration of why ECB mode is insecure:
- Take the Linux Tux penguin bitmap (raw pixel data)
- Encrypt with AES-ECB
- **Identical plaintext blocks → identical ciphertext blocks**
- Result: the penguin's outline is clearly visible in the "encrypted" image
- The white background encrypts to one repeating block, the black outline to another
- Demonstrates: ECB preserves patterns, provides NO semantic security
- **Always use CBC, CTR, or (best) GCM/CCM modes instead**

---

## 6. Diffie-Hellman Key Exchange

### Concrete Small-Number Example

**Public parameters:** p = 23 (prime), g = 5 (generator)

```
Alice picks secret:  a = 6
Alice computes:      A = g^a mod p = 5^6 mod 23 = 15625 mod 23 = 8
Alice sends A = 8 to Bob

Bob picks secret:    b = 15
Bob computes:        B = g^b mod p = 5^15 mod 23 = 30517578125 mod 23 = 19
Bob sends B = 19 to Alice

Alice computes:      s = B^a mod p = 19^6 mod 23 = 47045881 mod 23 = 2
Bob computes:        s = A^b mod p = 8^15 mod 23 = 35184372088832 mod 23 = 2

Shared secret: s = 2  ✓
```

**What an eavesdropper sees:** p=23, g=5, A=8, B=19
**What they need to find:** a such that 5^a ≡ 8 (mod 23) — the **discrete logarithm problem**

### Why Discrete Log is Hard

- With p=23, trivially solvable by trying all values (brute force 22 possibilities)
- With p = 2048-bit prime (~617 digits): no known classical algorithm runs in polynomial time
- **Best known:** General Number Field Sieve — sub-exponential but still infeasible for large p
- Real-world DH uses 2048-bit or 4096-bit primes
- Elliptic curve variant (ECDH) achieves equivalent security with 256-bit keys
- **Quantum threat:** Shor's algorithm solves discrete log in polynomial time — motivation for post-quantum crypto

### Real-world parameters:
- **RFC 3526 Group 14:** 2048-bit MODP group (widely used minimum)
- **RFC 7919:** Negotiated Finite Field DH groups for TLS 1.3
- **Curve25519:** 255-bit elliptic curve, ~128-bit security level, dominant in modern protocols

---

## 7. CSPRNG

### The `/dev/urandom` vs `/dev/random` Myth

**The myth:** "/dev/random is cryptographically secure, /dev/urandom is not"

**The reality:**
- Both use the **same CSPRNG** internally
- `/dev/random` historically blocked when estimated entropy was "low" — but this estimate was meaningless for a properly seeded CSPRNG
- **Since Linux kernel ~5.6 (2020):** `/dev/random` no longer blocks after initial seeding
- They are **functionally identical** on modern Linux
- `/dev/random` retains the ability to block only during early boot (before CSPRNG is seeded)

**Official man page (updated):** "The /dev/random interface is considered a legacy interface, and /dev/urandom is preferred and sufficient in all use cases, with the exception of applications which require randomness during early boot time."

### Modern Linux CSPRNG Architecture (kernel 5.6+)

1. **Entropy sources fed into pool:**
   - Hardware interrupts (timing jitter)
   - Disk I/O timing
   - User input events (keyboard, mouse)
   - RDRAND/RDSEED (Intel/AMD hardware RNG)
   - Jitter entropy (CPU execution time variations)

2. **Entropy extraction:**
   - **BLAKE2s** hash used to extract entropy from the input pool
   - Replaced the old SHA-1-based extraction (changed in kernel ~5.17)

3. **CSPRNG output:**
   - **ChaCha20-based** CSPRNG generates actual random bytes
   - Replaced the old SHA-1-based output generator
   - Per-CPU ChaCha20 instances (avoids lock contention)
   - Reseeded from entropy pool periodically

### RDRAND Instruction (Intel/AMD)

- Hardware random number generator on-die
- Provides random data from a NIST SP 800-90A compliant DRBG
- **Trust concern:** Black-box hardware, could theoretically be backdoored (NSA/Dual_EC_DRBG precedent)
- **Linux mitigation:** RDRAND is **one of many inputs** to the entropy pool, never used alone
- Even if RDRAND is compromised, it only *adds* entropy — cannot weaken other sources
- `CONFIG_RANDOM_TRUST_CPU=y` controls whether RDRAND alone can credit entropy at boot

### Key APIs by language:
- **C:** `randombytes_buf()` (libsodium), `getrandom(2)` syscall (Linux 3.17+)
- **Python:** `os.urandom()`, `secrets.token_bytes()` (Python 3.6+)
- **TypeScript:** `crypto.getRandomValues()` (Web Crypto), `crypto.randomBytes()` (Node)
- **Go:** `crypto/rand.Read()` (uses `/dev/urandom` or `getrandom(2)`)
- **Rust:** `OsRng` from `rand` crate, `getrandom` crate

---

## 8. Side-Channel Attacks

### Spectre Variant 1 — Bounds Check Bypass (CVE-2017-5753)

**Mechanism:**
```c
if (x < array1_size) {           // Branch predictor guesses TRUE
    y = array2[array1[x] * 256]; // Speculatively executes with attacker-controlled x
}
// CPU discovers x >= array1_size, rolls back architectural state
// BUT: array2 cache line loaded during speculation REMAINS in cache
// Attacker times access to array2[0..255*256] to determine array1[x]
```

**Step-by-step:**
1. Attacker trains branch predictor to expect the bounds check to pass (many valid x values)
2. Attacker provides out-of-bounds x (e.g., pointing to kernel memory)
3. CPU speculatively executes the load `array1[x]` (reads secret byte)
4. Secret byte used as index into `array2` — loads one cache line
5. Speculation squashed, but cache state persists
6. Attacker probes all 256 cache lines of `array2` — the fast one reveals the secret byte
7. Repeat for each byte of target memory

**Disclosed:** 2018-01-03 by Google Project Zero (Jann Horn) and independently by Paul Kocher et al.

### Cache Timing Attacks on AES (Bernstein 2005)

**Paper:** "Cache-timing attacks on AES" (2005-04-14, Daniel J. Bernstein)

**The vulnerability:** AES software implementations use **T-tables** (four 1KB lookup tables: T0, T1, T2, T3) mapping one byte to four bytes. These combine SubBytes, ShiftRows, and MixColumns into precomputed lookups.

**The attack:**
1. T-table lookups are indexed by `plaintext[i] XOR key[i]`
2. Different key bytes cause different cache line accesses
3. By measuring encryption time for many plaintexts, attacker builds a statistical profile
4. Compare timing profile against a reference server with known key
5. Correlations reveal which cache lines are accessed → reveals key bytes
6. **Full AES key recovery from network timing** demonstrated

**Significance:** First practical demonstration of **remote** cache timing attacks (not just local). Showed that even network jitter couldn't hide the signal with enough samples.

### Power Analysis on Smartcards

**Introduced:** 1998 by Paul Kocher, Joshua Jaffe, Benjamin Jun

**Simple Power Analysis (SPA):**
- Directly visible in power trace oscilloscope readings
- RSA square-and-multiply: squaring operations draw less power than multiply operations
- A multiply = bit "1" in private key exponent, square-only = bit "0"
- **Can extract entire RSA private key from a single power trace**

**Differential Power Analysis (DPA):**
- Statistical method using many traces
- Hypothesize key bit → predict intermediate values → correlate with measured power
- Averages out noise: works even when individual traces are too noisy for SPA
- Can extract AES keys from smartcards with ~1000 traces
- **Non-invasive, non-destructive, leaves no evidence of attack**

**Cost:** A basic power analysis setup costs ~$2,000–$5,000 (oscilloscope + current probe + target board). Professional setups with EM probes: $10,000–$50,000.

### Constant-Time Programming

**The rule:** Execution path and memory access pattern must be independent of secret data.

**Common violations and fixes:**

```c
/* BAD: secret-dependent branch */
if (secret_byte & 0x80) { do_something(); }

/* GOOD: branchless conditional */
mask = -(int)(secret_byte >> 7);  /* 0x00000000 or 0xFFFFFFFF */
result = (a & mask) | (b & ~mask);

/* BAD: secret-dependent array index (T-table AES) */
output = T[secret_byte];  /* cache line reveals secret_byte */

/* GOOD: bitsliced AES (no table lookups) */
/* Implement AES logic as boolean circuits operating on bits */

/* BAD: early-return comparison */
for (i = 0; i < len; i++) {
    if (a[i] != b[i]) return 0;  /* timing reveals position of first difference */
}

/* GOOD: constant-time comparison */
volatile unsigned char result = 0;
for (i = 0; i < len; i++) {
    result |= a[i] ^ b[i];
}
return result == 0;
```

**Language support:**
- C: `CRYPTO_memcmp()` (OpenSSL), `sodium_memcmp()` (libsodium)
- Go: `crypto/subtle.ConstantTimeCompare()`
- Rust: `ring` and `subtle` crate (`ConstantTimeEq` trait)
- Python: `hmac.compare_digest()`
- TypeScript: `crypto.timingSafeEqual()` (Node.js)

---

## 9. Password Hashing

### Why MD5/SHA for Passwords is Broken

**GPU hashing speeds (approximate, RTX 4090-class):**
| Algorithm | Hashes/second | Time for 8-char password |
|-----------|--------------|--------------------------|
| MD5 | **180 billion/sec** | Minutes |
| SHA-1 | ~70 billion/sec | Minutes |
| SHA-256 | ~22 billion/sec | Hours |
| bcrypt (cost=12) | ~30,000/sec | Weeks |
| scrypt (N=2^17) | ~1,000/sec | Months |
| Argon2id (128MB) | ~10/sec | Years |

The GPU advantage ratio: **MD5 is ~620,000x faster on GPU vs CPU.** Argon2id with 128MB memory is only ~1.5x faster on GPU vs CPU (memory bandwidth is the bottleneck, not compute).

### bcrypt

**Algorithm:** Blowfish-based adaptive hash (1999, Niels Provos & David Mazières)

**Cost factor:** The work factor parameter controls iterations as **2^cost**
- Cost 10: 2^10 = 1,024 iterations → ~65ms
- Cost 11: 2^11 = 2,048 iterations → ~130ms
- Cost 12: 2^12 = 4,096 iterations → **~250ms** (common production default)
- Cost 13: 2^13 = 8,192 iterations → ~510ms
- Cost 14: 2^14 = 16,384 iterations → ~1,015ms

**OWASP recommendation:** Tune so hashing takes 100–250ms. Currently cost **10 minimum**, 12 recommended.

**Limitations:**
- Max password length: **72 bytes** (silently truncates beyond that)
- Memory usage: fixed ~4KB (Blowfish state) — does NOT increase with cost
- No memory-hardness → vulnerable to FPGA/ASIC attacks (just not as bad as MD5/SHA)

**Output format:** `$2b$12$R9h/cIPz0gi.URNNX3kh2OPST9/PgBkqquzi.Ss7KIUgO2t0jWMUW`
- `$2b$` = algorithm version, `12` = cost factor, next 22 chars = base64-encoded salt, rest = hash

### scrypt (2009, Colin Percival)

**Key innovation:** Memory-hard — requires large amounts of RAM proportional to CPU time.

**Parameters:**
- **N (CPU/memory cost):** Must be power of 2. Doubles time AND memory when doubled.
- **r (block size):** Typically 8. Fine-tunes sequential memory read size.
- **p (parallelization):** Usually 1 for password hashing. Each thread needs N×r memory.

**Memory formula:** `128 × N × r` bytes

**Recommended settings:**
| Use case | N | r | p | Memory |
|----------|---|---|---|--------|
| Interactive login | 2^14 (16,384) | 8 | 1 | 16 MB |
| Sensitive/file encryption | 2^20 (1,048,576) | 8 | 1 | 1 GB |
| OWASP minimum | 2^17 (131,072) | 8 | 1 | 128 MB |

**Constraint:** `r × p < 2^30`

### Argon2id (2015, PHC winner)

**Designed by:** Alex Biryukov, Daniel Dinu, Dmitry Khovratovich (University of Luxembourg)
**Won:** Password Hashing Competition (PHC) in July 2015, selected from 24 candidates

**Three variants:**
- **Argon2d:** Data-dependent memory access. Best GPU/ASIC resistance. Vulnerable to side-channel.
- **Argon2i:** Data-independent memory access. Side-channel resistant. Weaker against GPU.
- **Argon2id:** **Hybrid — recommended for password hashing.** First half of first pass is Argon2i (side-channel resistant), rest is Argon2d (GPU resistant).

**Parameters:**
- **t (time cost):** Number of iterations over the memory. Minimum 1 for Argon2id.
- **m (memory cost):** Memory in KiB. OWASP minimum: 19 MiB. Recommended: 46 MiB+.
- **p (parallelism):** Number of threads. Typically 1–4.

**OWASP recommended configurations (in priority order):**
1. Argon2id, m=47104 (46 MiB), t=1, p=1
2. Argon2id, m=19456 (19 MiB), t=2, p=1
3. Argon2id, m=12288 (12 MiB), t=3, p=1
4. Argon2id, m=9216 (9 MiB), t=4, p=1
5. Argon2id, m=7168 (7 MiB), t=5, p=1

**Why Argon2id wins:**
- Memory-hard (like scrypt) — GPU parallelism limited by memory bandwidth
- Configurable parallelism (unlike bcrypt's fixed 4KB)
- Side-channel resistant (unlike pure Argon2d)
- Modern design incorporating lessons from bcrypt, scrypt, and PHC submissions
- RFC 9106 (2021) standardizes it

---

## Quick Reference: Chapter Mapping Suggestions

| Chapter | Topic | Key Demos |
|---------|-------|-----------|
| 1 | Why crypto matters | Heartbleed war story, plaintext password leaks |
| 2 | Symmetric encryption | AES internals, ECB penguin, modes of operation |
| 3 | Hash functions | SHA-256 internals, collision resistance, birthday attack |
| 4 | MACs & authenticated encryption | HMAC, AES-GCM, Poly1305, encrypt-then-MAC |
| 5 | Asymmetric crypto / DH | DH small-number example, discrete log, Curve25519 |
| 6 | Digital signatures | Ed25519, RSA signatures, certificate chains |
| 7 | Password hashing | bcrypt/scrypt/Argon2id comparison, GPU cracking speeds |
| 8 | Random numbers | CSPRNG, /dev/urandom myth, ChaCha20 DRBG |
| 9 | TLS 1.3 | Handshake flow, 1-RTT, 0-RTT risks, removed features |
| 10 | Signal Protocol | X3DH, Double Ratchet, forward secrecy, post-compromise |
| 11 | Side channels | Spectre, Bernstein AES timing, power analysis, constant-time |
| 12 | CVE war stories | Heartbleed, POODLE, Goto Fail, ROCA |
| 13 | Library APIs | libsodium, Python cryptography, Web Crypto, Go crypto, ring |
| 14 | Building the messenger | Putting it all together: X3DH + Double Ratchet + AES-GCM |
