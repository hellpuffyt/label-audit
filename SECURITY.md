# Security policy

## Supported versions

Only the latest commit on `main` receives fixes. There are no maintained release branches.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's advisory form:
https://github.com/hellpuffyt/label-audit/security/advisories/new

Do not open a public issue for a suspected vulnerability. Include the version or commit,
a minimal reproduction, and the impact you observed. You should receive an acknowledgement
within a few days; fixes are released as soon as they are verified.

## Scope

label-audit reads local CSV/JSON Lines datasets and prints a report; it makes no network calls, never modifies or deletes data, and never executes input content. Relevant reports include: input that executes code, crashes with a traceback instead of exit code 1, or produces a silently wrong statistic.
