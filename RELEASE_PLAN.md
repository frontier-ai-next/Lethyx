# Lethyx Release Plan

## Snapshot

`v0.6.0` is the first GitHub import of upstream `main`, commit
`c4a0015535adeba5bdce63e15b60ac7c431b7fe4`. The source package, tests, and docs
were renamed to `lethyx`; source-host pipeline and package-registry instructions
were removed. A GitHub Actions workflow now runs validation, tests, and wheel
packaging.

## Release readiness

The source snapshot has not yet been validated by this repository's GitHub CI.
The tag identifies the imported version; successful CI is still required before
claiming the release is verified.

## Remaining release work

- Reconcile preserved Lethyx benchmark harnesses with the current backend API.
- Confirm backend policy and optional Zvec support are documented consistently.
- Review public package metadata and examples under the `lethyx` namespace.
- Use the CI result as the release verification record.
