{ pkgs, elodin-sim }:

let
  # Use the Python package from elodin_sim flake
  elodinSimPython = elodin-sim.python;
  # Also include elodinPy directly to ensure it's available
  elodinPy = elodin-sim.elodinPy;
  # Include elodin and elodin-db binaries from elodin_sim flake
  elodinBin = elodin-sim.elodin;
  elodinDbBin = elodin-sim.elodin-db;
in

pkgs.mkShell {
  name = "sitl";
  packages = [
    elodinSimPython
    elodinPy
    elodinBin
    elodinDbBin
    pkgs.sitl-python
    # pkgs.elodin-python
    pkgs.gcc.cc.lib
    pkgs.gfortran.cc.lib
    pkgs.which
    pkgs.lesspipe
  ];
  shellHook = ''
    # Find the fsw root directory by traversing up from current directory
    # Look for flake.nix as a marker of the project root
    if [ -z "$ROOT_FSW_DIR" ]; then 
        export ROOT_FSW_DIR="$(pwd)";
        while [ "$ROOT_FSW_DIR" != "/" ]; do
            if [ -f "$ROOT_FSW_DIR/flake.nix" ]; then
                break;
            fi
            ROOT_FSW_DIR="$(dirname "$ROOT_FSW_DIR")";
        done
    fi

    if [ ! -f "$ROOT_FSW_DIR/flake.nix" ]; then
        echo "error: Expected 'flake.nix' in the fsw dir '$ROOT_FSW_DIR'" >&2;
        unset ROOT_FSW_DIR;
    fi
    if [ -z "$ELODIN_SIM" ]; then
       export ELODIN_SIM=$(realpath "$ROOT_FSW_DIR/external/elodin_sim");
    fi
    if [ -d "$ELODIN_SIM" ]; then
        # The elodin_sim Python wrapper already sets PYTHONPATH, but we want to ensure
        # the elodin_sim directory is also in PYTHONPATH for local modules
        if [ -z "${PYTHONPATH:-}" ]; then
            export PYTHONPATH="$ELODIN_SIM";
        else
            # Only add if not already present
            case ":$PYTHONPATH:" in
                *:"$ELODIN_SIM":*)
                    ;;
                *)
                    export PYTHONPATH="$ELODIN_SIM:$PYTHONPATH"
                    ;;
            esac
        fi
    else
        echo "error: No elodin_sim directory found. Expected to be in '$ROOT_FSW_DIR/external/elodin_sim' of fsw; PYTHONPATH unchanged." >&2;
        unset ELODIN_SIM;
    fi
    
    # Add fsw/shell or shell directory to PATH if found.
    if [ -d "$ROOT_FSW_DIR/shell" ]; then
      export PATH="$ROOT_FSW_DIR/shell:$PATH";
    else
      echo "warning: Could not find fsw directory. fsw/shell not added to path." >&2;
    fi
    
    echo "SITL environment ready"
    echo ""
    echo "To SITL:"
    echo "start_sim.sh sim_sitlXXX --sitl"
    echo "or"
    echo "SITL=1 python3 \$ELODIN_SIM/sim_samples/sim_FT09.py run --config config/config.toml --db-path sim_sitlXXX"
    echo "To HITL:"
    echo "start_sim.sh sim_hitlXXX --hitl"
    echo "or"
    echo "HITL=1 python3 \$ELODIN_SIM/sim_samples/sim_FT09.py run --db-path sim_hitlXXX"
    echo "To run sim:"
    echo "start_sim.sh simXXX --sim"
    echo "or"
    echo "python3 \$ELODIN_SIM/sim_samples/sim_FT09.py run --db-path simXXX"
    echo ""
    echo "Note: The Xs will be replaced by the next non-existent filename."
  '';
}

