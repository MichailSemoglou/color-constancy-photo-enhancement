# Security Policy

## Supported versions

Security fixes land on the latest release only. Older releases do not receive backports.

| Version | Supported |
| ------- | --------- |
| 1.3.x (latest) | Yes |
| < 1.3 | No |

## Reporting a vulnerability

Email m.semoglou@tongji.edu.cn with a description of the issue, the affected version, and a reproduction case. Do not open a public issue for an unpatched vulnerability.

Reports receive an acknowledgment within five working days. Accepted reports are fixed in the next patch release; the reporter is credited in the changelog unless they prefer otherwise.

## Scope notes

This package is an offline image-processing library and CLI. It holds no credentials, opens no network sockets, and runs no server. The practical attack surface is the decoding of untrusted input: image files passed to the CLI or the benchmark harness, and dataset CSV files consumed by the benchmark.

Vulnerabilities inside the media codecs themselves (OpenCV, libwebp, and similar bundled libraries) are tracked through the dependency floors in `pyproject.toml` and the `pip-audit` job in CI; please report those upstream. Weaknesses in how this package handles such input (missing size limits, unsafe file handling, CSV formula injection in generated reports) belong here.
