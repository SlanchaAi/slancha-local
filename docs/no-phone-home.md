# No phone-home — captured 2026-05-09

Default install: `slancha serve` with `SLANCHA_CLASSIFIER_KIND=local` (the default).
Routed 100 prompts through the proxy. Captured on the egress interface.

Method:

```bash
sudo tcpdump -i any -n 'not (host 127.0.0.1) and not (port 53) and not (port 5353)' \
    -w /tmp/slancha-egress.pcap &
TCPPID=$!

# Drive 100 prompts through the proxy
for i in $(seq 1 100); do
  curl -s http://127.0.0.1:8765/v1/chat/completions \
    -H 'content-type: application/json' \
    -d "{\"model\":\"auto\",\"messages\":[{\"role\":\"user\",\"content\":\"prompt $i: $(head -c 50 /dev/urandom | base64)\"}]}" \
    > /dev/null
done

sleep 2
kill $TCPPID
sudo tcpdump -r /tmp/slancha-egress.pcap | wc -l
```

Result: **0 packets captured** (excluding loopback and DNS).

The same command at `slancha-local doctor --capture` will tell you exactly what
the next request would egress before you run it.

## What changes if you opt in

`SLANCHA_CLASSIFIER_KIND=cloud SLANCHA_API_KEY=sk_... slancha serve` adds:

```
→ api.slancha.ai:443 (POST /v1/classify-routed)
  payload: {"embedding": [..512 floats..], "available_models": [...], ...}
```

The embedding is a 512-dim float32 vector, `len ≈ 4 KB`. By default,
`prompt: null` — the raw text is omitted.

Toggle `SLANCHA_SHARE_PROMPTS=true` to additionally include `prompt`. The
CLI prints a confirmation banner when this flag is set.

## Reproduce

```bash
slancha-local doctor --capture                # before sending
sudo tcpdump -i any -n ...                    # external verification
```
