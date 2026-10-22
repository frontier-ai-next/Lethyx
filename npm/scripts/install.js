"use strict";

const fs = require("fs");
const os = require("os");
const path = require("path");
const { spawnSync } = require("child_process");

function umMemHome() {
  return process.env.UM_MEM_HOME
    ? path.resolve(process.env.UM_MEM_HOME)
    : path.join(os.homedir(), ".um-mem");
}

function venvDir() {
  return process.env.UM_MEM_VENV
    ? path.resolve(process.env.UM_MEM_VENV)
    : path.join(umMemHome(), "python");
}

function venvPython() {
  return process.platform === "win32"
    ? path.join(venvDir(), "Scripts", "python.exe")
    : path.join(venvDir(), "bin", "python");
}

function engineBin() {
  return process.platform === "win32"
    ? path.join(venvDir(), "Scripts", "um-memory.exe")
    : path.join(venvDir(), "bin", "um-mem");
}

function mcpBin() {
  return process.platform === "win32"
    ? path.join(venvDir(), "Scripts", "um-memory-mcp.exe")
    : path.join(venvDir(), "bin", "um-memory-mcp");
}

function run(command, args, options) {
  const result = spawnSync(command, args, {
    stdio: "inherit",
    env: process.env,
    ...options,
  });
  if (result.error) {
    console.error(result.error.message);
    return 1;
  }
  return result.status == null ? 1 : result.status;
}

function findPython() {
  if (process.env.UM_MEM_PYTHON) return [process.env.UM_MEM_PYTHON];
  const candidates =
    process.platform === "win32"
      ? [["py", "-3.12"], ["py", "-3"], ["python"], ["python3"]]
      : [["python3.12"], ["python3"], ["python"]];
  for (const candidate of candidates) {
    const probe = spawnSync(candidate[0], [...candidate.slice(1), "-c", "import sys; print(sys.version_info[:2])"], {
      encoding: "utf8",
    });
    if (probe.status === 0 && /\(3, 1[2-9]\)/.test(probe.stdout || "")) {
      return candidate;
    }
  }
  return null;
}

function parseArgs(args) {
  const parsed = { from: "", semantic: null, force: false };
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === "--from" || arg === "--spec") parsed.from = args[++index] || "";
    else if (arg === "--semantic") parsed.semantic = true;
    else if (arg === "--no-semantic") parsed.semantic = false;
    else if (arg === "--force") parsed.force = true;
  }
  return parsed;
}

function defaultSpec(semantic) {
  return semantic ? "um-mem[mcp,semantic]" : "um-mem[mcp]";
}

function ensureEngine(rawArgs) {
  const options = parseArgs(rawArgs || []);
  if (fs.existsSync(engineBin()) && !options.force) {
    console.log(`UM-MEM is already installed at ${engineBin()}`);
    if (options.semantic === true) {
      console.log("To add the optional embedding runtime, run:");
      console.log("  um-memory install --semantic --force");
    } else {
      console.log("Use 'um-memory install --force' to reinstall or change extras.");
    }
    return 0;
  }
  const python = findPython();
  if (!python) {
    console.error("UM-MEM needs Python 3.12 or newer.");
    console.error("Install Python from https://www.python.org/downloads/ and run this again.");
    return 1;
  }
  fs.mkdirSync(umMemHome(), { recursive: true });
  if (!fs.existsSync(venvPython())) {
    console.log(`Creating a Python environment in ${venvDir()}`);
    const created = run(python[0], [...python.slice(1), "-m", "venv", venvDir()]);
    if (created !== 0) return created;
  }
  let spec = options.from || process.env.UM_MEM_PIP_SPEC || defaultSpec(options.semantic === true);
  if (options.semantic === true && options.from && !/\[/.test(spec)) {
    console.log("Note: --semantic is ignored with a custom --from spec; include the extra in the spec.");
  }
  console.log(`Installing ${spec}`);
  const installed = run(venvPython(), ["-m", "pip", "install", "--upgrade", spec]);
  if (installed !== 0) {
    console.error("\nThe Python package could not be installed.");
    console.error('For a source checkout use: um-memory install --from "C:\\\\path\\\\to\\\\UM-MEM"');
    return installed;
  }
  const marker = {
    installedAt: new Date().toISOString(),
    spec,
    venv: venvDir(),
    python: venvPython(),
    um-mem: engineBin(),
    mcp: mcpBin(),
  };
  fs.writeFileSync(path.join(umMemHome(), "npm.json"), JSON.stringify(marker, null, 2) + "\n");
  console.log(`\nUmMem installed. Engine: ${engineBin()}`);
  console.log("Next: um-memory setup");
  return 0;
}

module.exports = {
  ensureEngine,
  engineBin,
  findPython,
  umMemHome,
  mcpBin,
  venvDir,
  venvPython,
};
