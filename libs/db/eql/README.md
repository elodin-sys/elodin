# Elodin Query Language

## Introduction to Elodin-db and EQL
**Elodin-db** relies on a relational backend similar to the TimescaleDB extension for Postgres to store and query time-series signals.  
To simplify access to hierarchical, time-oriented data, we introduce **Elodin Query Language** (EQL).

EQL is a **lightweight query language dedicated to time-series signals**.  
It allows you to easily select components (e.g. `a.world_pos.x`, `rocket.fin_deflect[0]`), automatically handle time-based joins, and work with time windows (`.last("PT1m")`, `.first("PT1s")`).  
EQL also provides specialized signal-oriented functions such as `fft` or `fftfreq`.

## From EQL to SQL

EQL is not an independent engine: every EQL query is **translated into SQL** and then executed by Elodin-db. 
This process can be seen as *source-to-source compilation* (similar to TypeScript → JavaScript).

### First example
```eql
(drone.position.x, drone.velocity.y).last("5m")
```

would be translated into an equivalent SQL query:

```sql
select
  drone_position.drone_position[1] as "drone.position.x",
  drone_velocity.drone_velocity[2] as "drone.velocity.y"
from drone_position
join drone_velocity on drone_position.time = drone_velocity.time
where drone_position.time >= to_timestamp(<last_timestamp - 5 minutes>);
```

## Comprehensive Grammar

Everything listed here is supported by the parser & semantics (`Context::parse`).
Anything not listed is not part of the current language (as of **September 29, 2025**).

| Element                 | EQL form (examples)             | Meaning / Use                           | Notes (exact behavior) |
|--------------------------|---------------------------------|-----------------------------------------|-------------------------|
| **Identifier**           | `a`, `rocket`, `cow_0`          | Names an entity/component node           | Must start with a Unicode XID-start char; underscores/digits allowed after first char. |
| **Field access**         | `a.world_pos`, `rocket.fin_deflect` | Navigate hierarchical parts/components | Each `.` step moves to a child; at leaf it selects the component. |
| **Time field**           | `a.world_pos.time`              | Access the timestamp field of a component | Special semantic case → `Expr::Time`. |
| **Array element**        | `a.world_pos[0]`                | Select one element of a vector component | 0-based in EQL; converted to 1-based in SQL (`[index+1]`). Only valid on components. |
| **Tuple**                | `(a, b)`, `(a, b, c)`           | Select multiple expressions/columns      | Comma has the lowest precedence; top-level comma builds a tuple. |
| **Parentheses**          | `(expr)`                        | Grouping                                | Standard grouping of sub-expressions. |
| **Float literal**        | `12`, `-3`, `1.0`, `2.`         | Numeric literal                          | Grammar: optional `-`, digits, optional `.digits`. No exponent form (`1e3` not supported). |
| **String literal**       | `"5m"`, `"10s"`, `"text"`       | String value                             | No escapes; cannot contain `'` or `"` inside. Used by `last`/`first`. |
| **Binary operators**     | `+`, `-`, `*`, `/`              | Arithmetic in projections                | All four share one precedence level, left-associative. No comparisons/booleans. |
| **Formula call (generic)**| `expr.method(args)`             | Call a method on a receiver              | Parser accepts a (single) tuple of args due to comma precedence; semantics only support the formulas listed below (registered in `EqlFormula`). |
| **Formula: fft()**        | `a.world_pos.x.fft()`           | Signal FFT                               | Receiver must be an array access (scalar from vector). No args. Emits `fft(...)` in SQL (requires DB UDF). |
| **Formula: fftfreq()**    | `a.world_pos.time.fftfreq()`    | Frequency bins                           | Receiver must be `Time(...)` (component.time). No args. Emits `fftfreq(...)` in SQL (requires DB UDF). |
| **Formula: last("Δt")**   | `(expr).last("PT2S")`           | Time window (latest Δt)                  | Exactly one ISO-8601 string argument like `"PT2S"` (parsed via `jiff::Span`). Appends `WHERE time >= to_timestamp(last - Δt)`. |
| **Formula: first("Δt")**  | `(expr).first("PT5S")`          | Time window (earliest Δt)                | Exactly one ISO-8601 string argument like `"PT5S"`. Appends `WHERE time <= to_timestamp(earliest + Δt)`. |
| **Formula: norm()** | `a.world_pos.norm()` | Euclidean norm of a vector component | No args. Expands to sqrt(∑ elem*elem) over all vector elements; works on numeric vector components. |
| **Formula: atan2(y, x)** | `y.atan2(x)` | Two-argument arctangent | One arg. Computes atan2(y, x) in SQL. Receiver is y, argument is x. |
| **Formula: degrees(radians)** | `expr.degrees()` | Convert radians to degrees | No args. Converts angle from radians to degrees using PostgreSQL `degrees()` function. |
| **Formula: clip(value, min, max)** | `value.clip(min, max)` | Clamp value between min and max | Two args. Expands to `GREATEST(min, LEAST(value, max))` in SQL. |
| **Formula: sqrt(x)** | `expr.sqrt()` | Square root | No args. Computes square root using PostgreSQL `sqrt()` function. |
| **Formula: abs(x)** | `expr.abs()` | Absolute value | No args. Computes absolute value using PostgreSQL `abs()` function. |
| **Formula: arccos(x)** | `expr.arccos()` | Inverse cosine (arccosine) | No args. Computes arccosine using PostgreSQL `acos()` function. |
| **Formula: sign(x)** | `expr.sign()` | Sign function | No args. Returns -1, 0, or 1 based on the sign of x. Uses SQL CASE statement (DataFusion compatible). |
| **Whitespace**           | spaces / tabs / newlines        | Ignored separators                       | Grammar skips whitespace where sensible. |
| **Format string**        | `text ${expr} text`             | Parse into segments + embedded ASTs      | Separate entrypoint `fmt_string` returns `Vec<FmtNode>`. Raw `$` not allowed in plain segments. |
| **Not supported**        | comments                        | —                                       | No comment syntax. |
| **Not supported**        | exponent numbers `1e3`          | —                                       | Not recognized by `float()` rule. |
| **Not supported**        | string escapes / quotes inside  | —                                       | Strings cannot contain `'` or `"` inside; no escapes. |

### Method arity note (EQL ↔ SQL)

| Method   | EQL form                           | EQL args | SQL emitted                               | SQL args |
|----------|------------------------------------|---------:|-------------------------------------------|---------:|
| `fft`    | `a.world_pos.x.fft()`              |        0 | `fft(a_world_pos.a_world_pos[1])`         |        1 |
| `fftfreq`| `a.world_pos.time.fftfreq()`       |        0 | `fftfreq(a_world_pos.time)`               |        1 |
| `atan2`  | `a.velocity.y.atan2(a.velocity.x)`  |        1 | `atan2(a_velocity.a_velocity[2], a_velocity.a_velocity[1])` | 2 |
| `degrees`| `a.angle.degrees()`                 |        0 | `degrees(a_angle.a_angle)`                |        1 |
| `clip`   | `a.value.clip(0.0, 100.0)`         |        2 | `GREATEST(0.0, LEAST(a_value.a_value, 100.0))` | 3 |
| `sqrt`   | `a.value.sqrt()`                   |        0 | `sqrt(a_value.a_value)`                   |        1 |

## EQL → SQL Examples

> **Assumption:** component `a.world_pos` maps to table `a_world_pos(time, a_world_pos)`.

| EQL | SQL | Explanation |
|-----|-----|-------------|
| `(a.world_pos.x, a.world_pos.y, a.world_pos.z)` | `select a_world_pos.a_world_pos[1] as "a.world_pos.x", a_world_pos.a_world_pos[2] as "a.world_pos.y", a_world_pos.a_world_pos[3] as "a.world_pos.z" from a_world_pos;` | Select all 3 axes (x, y, z) of a 3D vector. |
| `(a.world_pos.x).last("5m")` | `select a_world_pos.a_world_pos[1] as "a.world_pos.x" from a_world_pos where a_world_pos.time >= to_timestamp(<last_timestamp - 5 minutes>);` | Select X axis over the last 5 minutes. |
| `(a.world_pos.time, a.world_pos[0])` | `select a_world_pos.time, a_world_pos.a_world_pos[1] as "a.world_pos.x" from a_world_pos;` | Select timestamp and X axis from the same component. |
| `a.world_pos.x * 9.81 + 1.0` | `select (a_world_pos.a_world_pos[1] * 9.81) + 1.0 from a_world_pos;` | Apply arithmetic directly on the X axis. |
| `(a.world_pos.x + 1.0) * 9.81` | `select (a_world_pos.a_world_pos[1] + 1.0) * 9.81 from a_world_pos;` | Same result as above — precedence is flat; parentheses here are just explicit. |
| `a.world_pos[2]` | `select a_world_pos.a_world_pos[3] as "a.world_pos.z" from a_world_pos;` | Select Z axis using array indexing (EQL 0-based → SQL 1-based). |
| `a.world_pos.time.fftfreq()` | `select fftfreq(a_world_pos.time) from a_world_pos;` | Frequency bins from the component’s time column (v0.14.2+). |
| `(a.world_pos.x.fft(), a.world_pos.time.fftfreq()).first("2s")` | `select fft(a_world_pos.a_world_pos[1]) as "fft(a.world_pos.x)", fftfreq(a_world_pos.time) as "fftfreq(a.world_pos.time)" from a_world_pos where a_world_pos.time <= to_timestamp(<earliest_timestamp + 2 seconds>);` | FFT of X axis with frequency bins, restricted to the first 2 seconds (v0.14.2+). |
| `(a.world_pos.x, b.velocity.y).last("5m")` | `select a_world_pos.a_world_pos[1] as "a.world_pos.x", b_velocity.b_velocity[2] as "b.velocity.y" from a_world_pos join b_velocity on a_world_pos.time = b_velocity.time where a_world_pos.time >= to_timestamp(<last_timestamp - 5 minutes>);` | Select X axis and Y velocity, joined on time, last 5 minutes. |
| `(a.world_pos.time, b.velocity.y)` | `select a_world_pos.time, b_velocity.b_velocity[2] as "b.velocity.y" from a_world_pos join b_velocity on a_world_pos.time = b_velocity.time;` | Select timestamp with velocity Y, joined on time. |
| `a.world_pos.x + b.velocity.y` | `select a_world_pos.a_world_pos[1] + b_velocity.b_velocity[2] from a_world_pos join b_velocity on a_world_pos.time = b_velocity.time;` | Arithmetic combining values from two components (implicit time join). |
| `(a.world_pos.time, a.world_pos.x, a.world_pos.y).last("30s")` | `select a_world_pos.time, a_world_pos.a_world_pos[1] as "a.world_pos.x", a_world_pos.a_world_pos[2] as "a.world_pos.y" from a_world_pos where a_world_pos.time >= to_timestamp(<last_timestamp - 30 seconds>);` | Select multiple columns over a 30-second window. |
| `(a.world_pos.x, a.world_pos.y).first("1h")` | `select a_world_pos.a_world_pos[1] as "a.world_pos.x", a_world_pos.a_world_pos[2] as "a.world_pos.y" from a_world_pos where a_world_pos.time <= to_timestamp(<earliest_timestamp + 1 hour>);` | Select first hour of data (X and Y axes). |
| `(a.world_pos.x, a.world_pos[1])` | `select a_world_pos.a_world_pos[1] as "a.world_pos.x", a_world_pos.a_world_pos[2] as "a.world_pos.y" from a_world_pos;` | Mix dot-field and array index to select X and Y. |
| `(a.world_pos, b.velocity, c.acceleration)` | `select a_world_pos.a_world_pos as "a.world_pos", b_velocity.b_velocity as "b.velocity", c_acceleration.c_acceleration as "c.acceleration" from a_world_pos join b_velocity on a_world_pos.time = b_velocity.time join c_acceleration on a_world_pos.time = c_acceleration.time;` | Select three full components with implicit time joins. |
| `a.velocity.y.atan2(a.velocity.x)` | `select atan2(a_velocity.a_velocity[2], a_velocity.a_velocity[1]) as "atan2(a.velocity.y, a.velocity.x)" from a_velocity;` | Compute two-argument arctangent (angle from y and x components). |
| `a.angle.degrees()` | `select degrees(a_angle.a_angle) as "degrees(a.angle)" from a_angle;` | Convert angle from radians to degrees. |
| `a.value.clip(0.0, 100.0)` | `select GREATEST(0.0, LEAST(a_value.a_value, 100.0)) as "clip(a.value)" from a_value;` | Clamp value between 0.0 and 100.0. |
| `a.value.sqrt()` | `select sqrt(a_value.a_value) as "sqrt(a.value)" from a_value;` | Compute square root. |
| `a.value.abs()` | `select abs(a_value.a_value) as "abs(a.value)" from a_value;` | Compute absolute value. |
| `a.value.arccos()` | `select acos(a_value.a_value) as "arccos(a.value)" from a_value;` | Compute arccosine (inverse cosine). |
| `a.value.sign()` | `select sign(a_value.a_value) as "sign(a.value)" from a_value;` | Compute sign: -1 if negative, 0 if zero, 1 if positive. |
| `((rocket.v_body[0] * -1.0) / rocket.v_body.norm().clip(0.000000001, 999999)).arccos().degrees() * (rocket.v_body[2] * -1.0).sign()` | `select (degrees(acos((rocket_v_body.rocket_v_body[1] * -1.0) / GREATEST(0.000000001, LEAST(sqrt(rocket_v_body.rocket_v_body[1] * rocket_v_body.rocket_v_body[1] + rocket_v_body.rocket_v_body[2] * rocket_v_body.rocket_v_body[2] + rocket_v_body.rocket_v_body[3] * rocket_v_body.rocket_v_body[3]), 999999)))) * sign((rocket_v_body.rocket_v_body[3] * -1.0)) as "..." from rocket_v_body;` | Compute angle of attack matching the component calculation: arccos(dot(u, [-1,0,0])/norm(u)) * -sign(w). |
| **Invalid shapes** | `a.world_pos.x.fft(1024) · a.world_pos.time.fftfreq(1) · a.world_pos.last(10)` | `fft`/`fftfreq` take no args; `last`/`first` require a string duration like `"10s"`. |
