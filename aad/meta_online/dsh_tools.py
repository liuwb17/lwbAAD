"""Browse public workspace files and edit policy drafts through local stdio MCP."""
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aad.io import digest, read_json, save_json
from aad.meta_online import COMPONENTS
from aad.meta_online.policy import validate

FILES = tuple(name + '.py' for name in COMPONENTS)


def prepare_workspace(root, evidence, original):
    """Materialize public evidence and all policy versions for agent-led inspection."""
    root = Path(root)
    workspace = root / 'workspace'
    paths = []

    def put(path, content):
        target = workspace / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')
        paths.append(path)

    def put_json(path, value):
        put(path, json.dumps(value, ensure_ascii=False, indent=2) + '\n')

    for key, value in evidence.items():
        if not key.isidentifier():
            raise ValueError('Invalid evidence section name')
        put_json(f'evidence/{key}.json', value)
    for index, version in enumerate(evidence['versions']):
        folder = f'history/{index:05d}'
        put_json(f'{folder}/version.json', version)
        for name, source in sources_at(evidence, version['revision']).items():
            put(f'{folder}/{name}', source)
    for name, source in original.items():
        if name not in FILES:
            raise ValueError('Invalid draft component')
        put(f'draft/{name}', source)
    put('README.txt', 'Inspect these folders in any order and analyze the public evidence yourself.\n'
        'evidence/: public problem, interfaces, settings, round effects, candidate records, decisions.\n'
        'history/: every policy version in chronological order; version.json identifies its revision.\n'
        'draft/: six editable Python components initialized from the selected parent policy.\n'
        'Use list_files, read_file (line ranges) and search_files (literal text) to explore.\n'
        'Use write_component to replace a draft file, then check_components.\n'
        'Only draft components are writable. No private results or credentials are exposed.\n')
    save_json(root / 'workspace_files.json', sorted(paths))


def read_draft(root):
    return {name: (Path(root) / 'workspace' / 'draft' / name).read_text(encoding='utf-8')
            for name in FILES}


def sources_at(evidence, revision):
    version = next(v for v in evidence['versions'] if v['revision'] == revision)
    sources = {name: evidence['source_blobs'][sha] for name, sha in version['files'].items()}
    if set(sources) != set(FILES) or any(digest(sources[n]) != sha for n, sha in version['files'].items()):
        raise ValueError('Policy history identity mismatch')
    return sources


def schema(name, description, properties, required=()):
    return {'name': name, 'description': description,
            'inputSchema': {'type': 'object', 'properties': properties,
                            'required': list(required), 'additionalProperties': False}}


TOOLS = [
    schema('list_files', 'List public workspace files, optionally under a relative directory prefix.',
           {'prefix': {'type': 'string'}}),
    schema('read_file', 'Read a public workspace file by relative path, with optional one-based line range.',
           {'path': {'type': 'string'}, 'start_line': {'type': 'integer', 'minimum': 1},
            'max_lines': {'type': 'integer', 'minimum': 1, 'maximum': 400}}, ['path']),
    schema('search_files', 'Find literal text in public workspace files; results include paths and line numbers.',
           {'text': {'type': 'string'}, 'prefix': {'type': 'string'},
            'offset': {'type': 'integer', 'minimum': 0}}, ['text']),
    schema('read_evidence', 'Read a top-level section of the supplied public history; empty section lists keys.',
           {'section': {'type': 'string'}}),
    schema('read_component', 'Read a current draft component or an immutable historical revision.',
           {'name': {'type': 'string', 'enum': list(FILES)}, 'revision': {'type': 'string'}}, ['name']),
    schema('write_component', 'Replace one of the six draft components with complete Python source.',
           {'name': {'type': 'string', 'enum': list(FILES)}, 'source': {'type': 'string'}}, ['name', 'source']),
    schema('check_components', 'Run the fixed syntax and restricted-Python checks on all six draft components. '
           'No model calls or ALE trial. The controller checks interfaces before adopting.', {}),
]


class PolicyTools:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.evidence = read_json(self.root / 'evidence.json')
        self.paths = read_json(self.root / 'workspace_files.json')

    def file_path(self, path):
        if path not in self.paths:
            raise ValueError('Path is not a public workspace file')
        workspace = self.root / 'workspace'
        target = workspace / path
        if not target.resolve().is_relative_to(workspace.resolve()):
            raise ValueError('Path escapes public workspace')
        # No symlink aliases, including links to another otherwise accessible file.
        if any(p.is_symlink() for p in (target, *target.parents) if p != self.root):
            raise ValueError('Linked workspace paths are not allowed')
        return target

    def call(self, name, args):
        definition = next((t for t in TOOLS if t['name'] == name), None)
        if definition is None or not isinstance(args, dict):
            raise ValueError('Unknown local policy tool')
        spec = definition['inputSchema']
        if set(args) - set(spec['properties']) or not set(spec['required']) <= set(args):
            raise ValueError('Invalid tool arguments')
        for key, value in args.items():
            field = spec['properties'][key]
            if field['type'] == 'string' and not isinstance(value, str):
                raise ValueError('Expected string argument')
            if field['type'] == 'integer' and (type(value) is not int or value < field['minimum']
                                              or value > field.get('maximum', value)):
                raise ValueError('Invalid integer argument')
        if name == 'list_files':
            return [path for path in self.paths if path.startswith(args.get('prefix', ''))]
        if name == 'read_file':
            lines = self.file_path(args['path']).read_text(encoding='utf-8').splitlines()
            start, count = args.get('start_line', 1), args.get('max_lines', 200)
            end, selected, size = start - 1, [], 0
            for line in lines[start - 1:start - 1 + count]:
                # Keep replies bounded even for giant JSON strings; never silently truncate a line.
                if size + len(line) > 65536:
                    if not selected:
                        raise ValueError('Line exceeds 64K characters; inspect the materialized source files')
                    break
                selected.append(line)
                size += len(line)
                end += 1
            return {'path': args['path'], 'start_line': start, 'total_lines': len(lines),
                    'text': '\n'.join(selected), 'next_line': end + 1 if end < len(lines) else None}
        if name == 'search_files':
            if not args['text']:
                raise ValueError('Search text must not be empty')
            offset, matched, results = args.get('offset', 0), 0, []
            for path in self.paths:
                if not path.startswith(args.get('prefix', '')):
                    continue
                for line_number, line in enumerate(self.file_path(path).read_text(encoding='utf-8').splitlines(), 1):
                    if args['text'] not in line:
                        continue
                    matched += 1
                    if matched <= offset:
                        continue
                    if len(results) == 100:
                        return {'matches': results, 'next_offset': offset + len(results)}
                    results.append({'path': path, 'line': line_number, 'text': line[:500],
                                    'preview_truncated': len(line) > 500})
            return {'matches': results, 'next_offset': None}
        if name == 'read_evidence':
            section = args.get('section', '')
            return sorted(self.evidence) if not section else self.evidence[section]
        draft = {n: self.file_path(f'draft/{n}').read_text(encoding='utf-8') for n in FILES}
        if name == 'check_components':
            validate(draft)
            return {'valid': True, 'checks': 'syntax, signatures, restricted Python; no ALE trial'}
        component = args['name']
        if component not in FILES:
            raise ValueError('Only six component names are allowed')
        if name == 'read_component':
            sources = sources_at(self.evidence, args['revision']) if args.get('revision') else draft
            return {'name': component, 'source': sources[component]}
        source = args['source'].replace('\r\n', '\n').replace('\r', '\n')
        if len(source.encode('utf-8')) > 65536:
            raise ValueError('Component exceeds 64 KB')
        draft[component] = source
        self.file_path(f'draft/{component}').write_text(source, encoding='utf-8')
        save_json(self.root / 'draft.json', draft)
        return {'name': component, 'sha256': digest(source)}

    def handle(self, request):
        method, params = request.get('method'), request.get('params') or {}
        if method == 'initialize':
            return {'protocolVersion': '2024-11-05', 'capabilities': {'tools': {}},
                    'serverInfo': {'name': 'rsi-policy', 'version': '1.0.0'},
                    'instructions': 'Only supplied public history and six draft components are accessible. '
                    'No network research, shell, model calls or private evaluation.'}
        if method == 'ping':
            return {}
        if method == 'tools/list':
            return {'tools': TOOLS}
        if method in ('resources/list', 'resources/templates/list'):
            return {'resources' if method == 'resources/list' else 'resourceTemplates': []}
        if method == 'tools/call':
            try:
                result = self.call(params['name'], params.get('arguments', {}))
                return {'content': [{'type': 'text', 'text': json.dumps(result, ensure_ascii=False)}]}
            except (ValueError, KeyError, StopIteration, TypeError) as error:
                return {'isError': True, 'content': [{'type': 'text', 'text': str(error)[:2000]}]}
        raise ValueError('Unsupported MCP method')


def main():
    server = PolicyTools(sys.argv[1])
    for line in sys.stdin:
        request = json.loads(line)
        if 'id' not in request:
            continue
        response = {'jsonrpc': '2.0', 'id': request['id']}
        try:
            response['result'] = server.handle(request)
        except (ValueError, KeyError, TypeError):
            response['error'] = {'code': -32601, 'message': 'Unsupported request'}
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
