#!/usr/bin/env python3
"""Make a private session config without outgoing automatic SHM allocation.

The vendor file (including endpoints/authentication) is copied, never edited.
The data session changes only outgoing automatic SHM allocation. An optional
firmware CLI session also selects client mode for short-lived routed RPCs.
The generated file can contain credentials: mode 0600, never include in Git.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


def patch_config(source: str, *, session_mode=None) -> str:
    if session_mode not in (None, 'client'):
        raise ValueError('Only the firmware client session override is supported')
    # Tokenize strings/comments too, so a comment or password cannot masquerade
    # as a configuration key. Retain original byte positions for the one edit.
    lexer = re.compile(
        r'\s+|//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"'
        r"|'(?:\\.|[^'\\])*'|[{}\[\]:,]|[^\s{}\[\]:,]+"
    )
    tokens = [m for m in lexer.finditer(source)
              if not m.group().isspace() and not m.group().startswith(('//', '/*'))]
    targets = {('transport', 'shared_memory', 'transport_optimization', 'enabled'): 'false'}
    if session_mode is not None:
        targets[('mode',)] = json.dumps(session_mode)
    matches = {path: [] for path in targets}

    def value(index, path):
        token = tokens[index].group()
        if token == '{':
            index += 1
            while tokens[index].group() != '}':
                key = tokens[index].group().strip('"\'')
                if tokens[index + 1].group() != ':':
                    raise ValueError('Expected configuration key/value')
                index = value(index + 2, (*path, key))
                if tokens[index].group() == ',':
                    index += 1
                elif tokens[index].group() != '}':
                    raise ValueError('Expected configuration separator')
            return index + 1
        if token == '[':
            index += 1
            while tokens[index].group() != ']':
                index = value(index, (*path, '[]'))
                if tokens[index].group() == ',':
                    index += 1
                elif tokens[index].group() != ']':
                    raise ValueError('Expected array separator')
            return index + 1
        if path in targets:
            if path == ('mode',):
                if token.strip('"\'') not in ('peer', 'client', 'router'):
                    raise ValueError('Invalid top-level session mode')
            elif token not in ('true', 'false'):
                raise ValueError('Automatic SHM setting must be boolean')
            matches[path].append(tokens[index])
        return index + 1

    try:
        consumed = value(0, ())
    except IndexError as exc:
        raise ValueError('Incomplete session configuration') from exc
    if consumed != len(tokens) or any(len(found) != 1 for found in matches.values()):
        raise ValueError('Require exactly one outgoing automatic SHM setting and selected top-level mode')
    edits = [(found[0], targets[path]) for path, found in matches.items()]
    for match, replacement in sorted(edits, key=lambda x:x[0].start(), reverse=True):
        source = source[:match.start()] + replacement + source[match.end():]
    return source


def prepare(source: Path, output: Path, *, session_mode=None):
    if source.resolve() == output.resolve():
        raise ValueError('Cannot overwrite vendor session configuration')
    original = source.read_text()
    patched = patch_config(original, session_mode=session_mode)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=output.parent, delete=False) as f:
        temporary = Path(f.name)
        try:
            os.chmod(temporary, 0o600)
            f.write(patched)
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = dict(source=str(source), output=str(output),
                   source_sha256=hashlib.sha256(original.encode()).hexdigest(),
                   output_sha256=hashlib.sha256(patched.encode()).hexdigest(),
                   changed_setting='transport.shared_memory.transport_optimization.enabled',
                   value=False)
    if session_mode is not None:
        receipt['session_mode'] = session_mode
    output.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cli-output', type=Path,
                        help='Separate private client-mode config for firmware action RPCs')
    args = parser.parse_args()
    if args.cli_output is not None and args.cli_output.resolve() in (args.source.resolve(), args.output.resolve()):
        parser.error('--cli-output must differ from source and the data-session output')
    prepare(args.source, args.output)
    if args.cli_output is not None:
        prepare(args.source, args.cli_output, session_mode='client')
