# Spark smoke status

> Spark side fills this in after running `scripts/spark_smoke.sh`.
> Mac side reads it on next session start.

## Latest

```
status: not yet run
last_attempt: never
result: -
notes: -
```

## Format for new entries (prepend, don't overwrite)

```
2026-MM-DDTHH:MM:SSZ [smoke] <pass | fail | partial> · 176 tests <green | red> · proxy <up | down> · decision-trace <observed | absent> · backend <ollama | none>
notes: free text — what worked, what didn't, what surprised you
```
