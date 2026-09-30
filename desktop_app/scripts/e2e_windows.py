"""Exercise the real installed Tauri/WebView2 app (no mocked backend).

Requires Playwright in the test environment, not on end-user machines.
Uses isolated WebView2 and replay data plus remote debugging for this process.
CUBESPRITE_DATA_DIR isolates the real native backend's storage without changing
the user's Known Folders or replacing the application's IPC transport.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request

from playwright.sync_api import sync_playwright, expect


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=9722)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env['WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS'] = f'--remote-debugging-port={args.port}'
    env['WEBVIEW2_USER_DATA_FOLDER'] = str(output / 'webview')
    env['CUBESPRITE_DATA_DIR'] = str(output / 'app-data')
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    process = subprocess.Popen([str(args.exe.resolve())], env=env, startupinfo=startup)
    checks = []
    errors = []
    passed = False
    page = None
    def record(name, **details):
        checks.append({'check': name, **details})
        print(name, flush=True)

    try:
        endpoint = f'http://127.0.0.1:{args.port}'
        deadline = time.monotonic() + 60
        while True:
            try:
                urllib.request.urlopen(endpoint + '/json/version', timeout=1).close()
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise RuntimeError('Installed app did not expose its test WebView2 endpoint')
                time.sleep(.25)
        with sync_playwright() as playwright:
            browser = playwright.chromium.connect_over_cdp(endpoint)
            context = browser.contexts[0]
            deadline = time.monotonic() + 30
            while not context.pages and time.monotonic() < deadline:
                time.sleep(.1)
            page = context.pages[0]
            page.set_default_timeout(30000)
            page.set_viewport_size({'width': 1440, 'height': 900})
            page.on('pageerror', lambda exc: errors.append(str(exc)))
            expect(page.get_by_role('button', name='EN', exact=True)).to_be_visible()
            page.get_by_role('button', name='EN', exact=True).click()
            expect(page.get_by_role('button', name='Player vs Player', exact=True)).to_be_visible()
            page.evaluate('''() => {
              window.testRpc = async (command, params = {}) => {
                const api = window.__TAURI_INTERNALS__;
                const id = 'e2e-' + crypto.randomUUID();
                let eventId;
                return new Promise(async (resolve, reject) => {
                  const timer = setTimeout(() => reject(new Error('RPC timeout: '+command)), 120000);
                  const handler = api.transformCallback(event => {
                    for (const line of String(event.payload).trim().split(/\\r?\\n/)) {
                      let value; try { value = JSON.parse(line); } catch { continue; }
                      if (value.id !== id) continue;
                      clearTimeout(timer);
                      api.invoke('plugin:event|unlisten', {event:'sidecar-stdout', eventId});
                      if (value.ok) resolve(value.result); else reject(new Error(JSON.stringify(value.error)));
                    }
                  });
                  eventId = await api.invoke('plugin:event|listen', {event:'sidecar-stdout',target:{kind:'Any'},handler});
                  await api.invoke('write_sidecar', {line:JSON.stringify({v:1,type:'request',id,command,params})});
                });
              };
            }''')
            def rpc(command, params=None):
                return page.evaluate('([c,p]) => window.testRpc(c,p)', [command, params or {}])
            def state():
                return rpc('game.state')
            def click(name):
                page.get_by_role('button', name=name, exact=True).click()
            def move(layer, row, col):
                before = state()['revision']
                page.locator(f'button[aria-label^="F{layer+1}, {row+1}, {col+1}:"]').click()
                deadline = time.monotonic() + 20
                while state()['revision'] == before:
                    if time.monotonic() > deadline:
                        raise AssertionError('Move did not reach the real backend')
                    time.sleep(.05)
            def select_rule(rule):
                page.locator(f'input[name="game-rule"][value="{rule}"]').check(force=True)
            def assert_viewport_fit(label, selectors):
                metrics = page.evaluate('''selectors => {
                  const viewport = {width: window.innerWidth, height: window.innerHeight};
                  const documentSize = {
                    width: Math.max(document.documentElement.scrollWidth, document.body.scrollWidth),
                    height: Math.max(document.documentElement.scrollHeight, document.body.scrollHeight)
                  };
                  const boxes = Object.fromEntries(selectors.map(selector => {
                    const element = document.querySelector(selector);
                    if (!element) throw new Error('Missing layout element: ' + selector);
                    const rect = element.getBoundingClientRect();
                    return [selector, {left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom}];
                  }));
                  return {viewport, documentSize, boxes};
                }''', selectors)
                viewport = metrics['viewport']
                document_size = metrics['documentSize']
                assert document_size['width'] <= viewport['width'] + 1, (label, metrics)
                assert document_size['height'] <= viewport['height'] + 1, (label, metrics)
                for selector, box in metrics['boxes'].items():
                    assert box['left'] >= -1 and box['top'] >= -1, (label, selector, metrics)
                    assert box['right'] <= viewport['width'] + 1, (label, selector, metrics)
                    assert box['bottom'] <= viewport['height'] + 1, (label, selector, metrics)
                record('Responsive ' + label, metrics=metrics)
                return metrics
            def assert_menu_panel_fit(label):
                metrics = assert_viewport_fit(label, ['.menu-screen', '.primary-menu', '.side-picker'])
                primary = metrics['boxes']['.primary-menu']
                panel = metrics['boxes']['.side-picker']
                overlap_width = min(primary['right'], panel['right']) - max(primary['left'], panel['left'])
                overlap_height = min(primary['bottom'], panel['bottom']) - max(primary['top'], panel['top'])
                assert overlap_width <= 1 or overlap_height <= 1, (label, metrics)
            viewports = [(1536, 792), (1100, 720), (1280, 720), (1440, 900), (1920, 1080)]
            init = rpc('system.initialize')
            existing_replay_ids = {item['id'] for item in rpc('replay.list')['replays']}
            assert init['backend_version'] == '0.2.0-alpha.1', init['backend_version']
            assert init['models'][0]['id'] == 'cubesprite_v4_flash_preview1'
            assert init['models'][0]['available']
            assert init['mcts_options'] == [16, 32, 64, 128, 256, 512, 1024]
            page.screenshot(path=str(output/'menu-en.png'))
            assert not re.search(r'0\.[12]\.\d', page.locator('body').inner_text())
            record('Installed app boots with V4 first and no visible app version')

            for width, height in viewports:
                page.set_viewport_size({'width': width, 'height': height})
                suffix = f'{width}x{height}'
                assert_viewport_fit('menu ' + suffix, ['.menu-screen', '.primary-menu', '.intelligence-box'])
                page.screenshot(path=str(output/f'responsive-menu-{suffix}.png'))
                click('Player vs AI')
                expect(page.locator('.side-picker')).to_be_visible()
                assert_menu_panel_fit('side picker ' + suffix)
                page.screenshot(path=str(output/f'responsive-side-{suffix}.png'))
                page.locator('.side-picker .side-picker-heading button').click()
                click('Replay')
                expect(page.locator('.replay-picker')).to_be_visible()
                assert_menu_panel_fit('replay picker ' + suffix)
                page.screenshot(path=str(output/f'responsive-replay-picker-{suffix}.png'))
                page.locator('.replay-picker .side-picker-heading button').click()
            for screen_name, button_name in [('ai-settings', 'AI Settings'), ('settings', 'Settings'), ('instructions', 'Instructions')]:
                click(button_name)
                for width, height in viewports:
                    page.set_viewport_size({'width': width, 'height': height})
                    suffix = f'{width}x{height}'
                    assert_viewport_fit(screen_name + ' ' + suffix, ['.page-shell', '.page-content', '.back-button'])
                    page.screenshot(path=str(output/f'responsive-{screen_name}-{suffix}.png'))
                click('Back to main menu')
            page.set_viewport_size({'width': 1440, 'height': 900})
            expect(page.get_by_role('slider', name='AI Intelligence')).to_be_visible()
            expect(page.get_by_role('button', name='Advance', exact=True)).to_be_visible()
            click('中')
            expect(page.get_by_role('slider', name='AI 智能度')).to_be_visible()
            expect(page.get_by_role('button', name='高级设置', exact=True)).to_be_visible()
            expect(page.locator('label:has(input[name="game-rule"][value="classic"]) span')).to_have_text('经典')
            page.screenshot(path=str(output/'responsive-menu-zh-1440x900.png'))
            click('EN')
            assert_viewport_fit('menu restored 1440x900', ['.menu-screen', '.primary-menu'])

            click('AI Settings')
            expect(page.locator('input[name="model-combat"]:checked')).to_have_value('cubesprite_v4_flash_preview1')
            expect(page.locator('.role-combat .parameter-block output')).to_have_text('256')
            expect(page.locator('.role-combat input[aria-label="Temperature"]')).to_have_value('0.4')
            page.locator('input[name="model-combat"][value="v2.2_balance"]').check(force=True)
            click('Back to main menu')
            select_rule('p1_vertical_ignored')
            expect(page.locator('.toast.visible')).to_contain_text('CubeSprite V4 Flash')
            click('AI Settings')
            expect(page.locator('input[name="model-combat"]:checked')).to_have_value('cubesprite_v4_flash_preview1')
            page.locator('input[name="model-combat"][value="v2.2_balance"]').check(force=True)
            expect(page.locator('.toast.visible')).to_contain_text('Classic')
            click('Back to main menu')
            expect(page.locator('input[name="game-rule"]:checked')).to_have_value('classic')
            record('Both model/rule selection orders show the compatibility banner and route correctly')

            slider = page.get_by_role('slider', name='AI Intelligence')
            effort_names = ['Starter', 'Swift', 'Balanced', 'Focused', 'Deep', 'Master', 'Maximum']
            for index, (sims, effort_name) in enumerate(zip([16, 32, 64, 128, 256, 512, 1024], effort_names)):
                slider.focus()
                slider.press('Home' if index == 0 else 'ArrowRight')
                expect(slider).to_have_value(str(index))
                expect(slider).to_have_attribute('aria-valuetext', f'{effort_name}, {sims} MCTS')
                click('Advance')
                expect(page.locator('.role-combat .parameter-block output')).to_have_text(str(sims))
                expect(page.locator('.role-combat input[aria-label="Temperature"]')).to_have_value('0.5')
                click('Back to main menu')
            slider_box = slider.bounding_box()
            assert slider_box
            for index, x in [(0, 1), (3, slider_box['width'] / 2), (6, slider_box['width'] - 1)]:
                slider.click(position={'x': x, 'y': slider_box['height'] / 2})
                expect(slider).to_have_value(str(index))
            slider.focus()
            slider.press('Home')
            record('Seven intelligence presets map to 16–1024 simulations at temperature 0.5')
            if page.locator('.toast.visible button').count():
                page.locator('.toast.visible button').click()

            for rule in init['rule_ids']:
                select_rule(rule)
                click('Player vs Player')
                expect(page.locator('.game-screen')).to_be_visible()
                assert state()['rule_id'] == rule
                if rule == 'classic':
                    for width, height in [(1100, 720), (1280, 720), (1536, 792)]:
                        page.set_viewport_size({'width': width, 'height': height})
                        suffix = f'{width}x{height}'
                        assert_viewport_fit('game 2D ' + suffix, ['.game-screen', '.game-topbar', '.game-workspace', '.board-area', '.game-functions'])
                        cell = page.locator('.board-cell').first.bounding_box()
                        assert cell and cell['width'] >= 37 and cell['height'] >= 37, cell
                        record('Readable board cells ' + suffix, width=cell['width'], height=cell['height'])
                        page.screenshot(path=str(output/f'responsive-game-2d-{suffix}.png'))
                        click('Switch to 3D')
                        expect(page.locator('canvas')).to_be_visible()
                        assert_viewport_fit('game 3D ' + suffix, ['.game-screen', '.game-topbar', '.game-workspace', '.game-functions'])
                        page.screenshot(path=str(output/f'responsive-game-3d-{suffix}.png'))
                        click('Switch to 2D')
                    page.set_viewport_size({'width': 1440, 'height': 900})
                moves = [(0,0,0),(0,4,4),(1,0,0),(0,4,3),(2,0,0),(0,3,4)]
                for coords in moves:
                    move(*coords)
                if rule == 'p1_vertical_forbidden':
                    expect(page.locator('button[aria-label^="F4, 1, 1:"]')).to_be_disabled()
                    assert all(m['action'] != 75 for m in state()['legal_moves'])
                    page.screenshot(path=str(output/'forbidden-board.png'))
                    click('Switch to 3D')
                    expect(page.locator('canvas')).to_be_visible()
                    page.screenshot(path=str(output/'forbidden-board-3d.png'))
                    click('Switch to 2D')
                else:
                    move(3,0,0)
                    expected = 'playing' if rule in ('p1_vertical_ignored','p1_vertical_and_layer0_ignored') else 'won'
                    assert state()['status'] == expected, (rule,state())
                click('Exit')
            record('All five rules: legal UI moves, vertical wins/ignored lines, forbidden landing disabled')

            for rule in ['classic','p1_layer0_ignored','p1_vertical_and_layer0_ignored']:
                select_rule(rule)
                click('Player vs Player')
                for coords in [(0,0,0),(0,4,0),(0,0,1),(0,4,2),(0,0,2),(0,3,4),(0,0,3)]:
                    move(*coords)
                assert state()['status'] == ('won' if rule == 'classic' else 'playing')
                click('Exit')
            record('First-floor line follows Classic versus ignored rules')

            select_rule('p1_vertical_forbidden')
            click('Player vs Player')
            move(0,2,2)
            before = state()
            click('Instructions')
            expect(page.get_by_text('Quick start', exact=True)).to_have_count(0)
            page.screenshot(path=str(output/'instructions.png'))
            click('Back')
            assert state() == before
            click('AI Settings')
            page.locator('input[name="model-combat"][value="v2.2_balance"]').click(force=True)
            expect(page.locator('input[name="model-combat"]:checked')).to_have_value('cubesprite_v4_flash_preview1')
            expect(page.locator('.toast.visible')).to_be_visible()
            click('Back')
            assert state() == before
            click('Save Replay')
            expect(page.locator('.toast.visible')).to_contain_text('Replay saved')
            saved = rpc('replay.list')['replays'][0]
            exported = rpc('replay.export', {'replay_id': saved['id'], 'expected_fingerprint': saved['fingerprint']})
            doc = json.loads(exported['content'])
            assert doc['protocol_version'] == 2 and doc['rule_id'] == 'p1_vertical_forbidden'
            (output/'saved-game.c4replay.json').write_text(exported['content'], encoding='utf-8')
            click('Exit')
            record('Instructions/AI Settings return to exact live state; incompatible switch rejected; save emits v2')

            select_rule('p1_layer0_ignored')
            click('Player vs AI')
            click('Blue · Second')
            deadline = time.monotonic() + 120
            while state()['move_count'] < 1:
                if time.monotonic() > deadline:
                    raise AssertionError('Bundled V4 did not play its opening')
                time.sleep(.2)
            expect(page.locator('.board-cell:not(:disabled)').first).to_be_enabled()
            assert state()['rule_id'] == 'p1_layer0_ignored'
            page.screenshot(path=str(output/'ai-game.png'))
            click('Exit')
            record('Bundled ONNX V4 plays a real AI opening under a nonclassic rule')

            click('Replay')
            page.locator('input[type="file"]').set_input_files(str(args.sample.resolve()))
            sample = json.loads(args.sample.read_text(encoding='utf-8'))
            page.get_by_role('button', name='Open ' + sample['name'], exact=True).click()
            expect(page.locator('.replay-screen')).to_be_visible()
            page.set_viewport_size({'width': 1100, 'height': 720})
            assert_viewport_fit('replay 1100x720', ['.replay-screen', '.game-topbar', '.replay-workspace-shell', '.game-functions'])
            page.screenshot(path=str(output/'responsive-replay-1100x720.png'))
            for _ in sample['turns']:
                click('Next')
            expect(page.get_by_role('button', name='Next', exact=True)).to_be_disabled()
            page.screenshot(path=str(output/'training-replay-end.png'))
            click('From Start')
            click('Next')
            click('Continue Here')
            expect(page.locator('.continue-dialog')).to_be_visible()
            assert_viewport_fit('continue dialog 1100x720', ['.dialog-backdrop', '.continue-dialog', '.continue-options'])
            page.screenshot(path=str(output/'responsive-continue-dialog-1100x720.png'))
            page.locator('.continue-options button').first.click()
            expect(page.locator('.game-screen:not(.replay-screen)')).to_be_visible()
            assert state()['move_count'] == 1 and state()['rule_id'] == sample['rule_id']
            page.set_viewport_size({'width': 1440, 'height': 900})
            click('Exit')
            click('中')
            if page.locator('.toast.visible button').count():
                page.locator('.toast.visible button').click()
            page.screenshot(path=str(output/'menu-zh.png'))
            click('对局回放')
            expect(page.locator('.replay-picker')).to_be_visible()
            assert_menu_panel_fit('replay picker Chinese 1440x900')
            page.screenshot(path=str(output/'responsive-replay-picker-zh.png'))
            page.locator('.replay-picker .side-picker-heading button').click()
            record('Training sample imports, plays to terminal frame, and continues from selected position')
            assert not errors, errors
            for item in rpc('replay.list')['replays']:
                if item['id'] in {saved['id'], sample['id']} and item['id'] not in existing_replay_ids:
                    rpc('replay.delete', {'id': item['id'], 'expected_fingerprint': item['fingerprint']})
            record('No browser page errors')
            passed = True
    except Exception:
        if page:
            try:
                page.screenshot(path=str(output/'failure.png'))
                (output/'failure-dom.txt').write_text(page.locator('body').inner_text(), encoding='utf-8')
            except Exception:
                pass
        raise
    finally:
        (output/'results.json').write_text(json.dumps({'passed': passed, 'executable': str(args.exe.resolve()), 'checks': checks, 'page_errors': errors}, ensure_ascii=False, indent=2), encoding='utf-8')
        if process.poll() is None:
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True)


if __name__ == '__main__':
    main()
