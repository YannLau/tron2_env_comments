## Summary

- What public problem does this change solve?
- What user-visible behaviour changes?

## Scope and risk

- [ ] The change stays within the public scope in `README.md` and `CONTRIBUTING.md`.
- [ ] It contains no credentials, private endpoints, robot addresses, camera serial
      numbers, customer/site data, logs, captures, datasets, or model weights.
- [ ] I described any effect on robot commands, joint mapping, rates, force/speed,
      transport messages, observations, safety behaviour, or emergency-stop integration.
- [ ] Changes affecting real-robot or security behaviour have the required owner review.

## Third-party content

- [ ] No third-party code or assets were added or changed; or I updated `NOTICE`
      with the source, version/commit, license, local path, and modifications.

## Verification

- [ ] `python -m compileall -q src tests examples`
- [ ] `python -m pytest -q`
- [ ] `python examples/mock_quickstart.py`
- [ ] `python -m build`
- [ ] No real robot, Bridge, or camera was contacted by the public CI checks.

## Authorised hardware validation (if applicable)

Describe the approved test scope and outcome without including private environment
details. Write `Not run` when the change was validated only with mocks.
