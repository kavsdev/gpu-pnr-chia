import sys
import re

def_file = sys.argv[1]
pl_file = sys.argv[2]
out_file = sys.argv[3]

# 1. Read PL file
# format: node_name X Y : ORIENT
pl_coords = {}
with open(pl_file, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('UCLA') or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) >= 5 and parts[3] == ':':
            name = parts[0]
            x = parts[1]
            y = parts[2]
            orient = parts[4]
            # FS means FN in DEF (flipped south) or something?
            # Bookshelf orientations: N, S, E, W, FN, FS, FE, FW
            # DEF orientations: N, S, E, W, FN, FS, FE, FW
            pl_coords[name] = (x, y, orient)

print(f"Loaded {len(pl_coords)} coordinates from {pl_file}")

# 2. Map DEF names to PL names, and update DEF
# State machine for DEF
with open(def_file, 'r') as f:
    def_lines = f.readlines()

in_components = False
out_lines = []

i = 0
updated = 0
not_found = 0

def get_pl_name(def_name):
    # Remove backslashes
    name = def_name.replace('\\', '')
    # Replace leading _ with n_
    if name.startswith('_'):
        name = 'n' + name
    return name

while i < len(def_lines):
    line = def_lines[i]
    if line.startswith('COMPONENTS '):
        in_components = True
        out_lines.append(line)
        i += 1
        continue
    
    if in_components and line.startswith('END COMPONENTS'):
        in_components = False
        out_lines.append(line)
        i += 1
        continue
    
    if in_components and line.strip().startswith('- '):
        # We are at the start of a component. It might span multiple lines, ending with ';'
        comp_str = line
        while ';' not in comp_str:
            i += 1
            comp_str += def_lines[i]
        
        # Parse comp_str
        # - NAME MACRO_NAME ... + PLACED ( X Y ) ORIENT ;
        # Use regex to find the name
        m = re.match(r'^\s*-\s+(\S+)', comp_str)
        if m:
            def_name = m.group(1)
            pl_name = get_pl_name(def_name)
            if pl_name in pl_coords:
                x, y, orient = pl_coords[pl_name]
                # Replace the + PLACED ( ... ) ORIENT ;
                # or + FIXED ( ... ) ORIENT ;
                # Regex to match: \+ (PLACED|FIXED) \( \d+ \d+ \) \S+
                new_status = f"+ PLACED ( {x} {y} ) {orient}"
                comp_str = re.sub(r'\+\s+(PLACED|FIXED)\s+\(\s+\d+\s+\d+\s+\)\s+\S+', new_status, comp_str)
                updated += 1
            else:
                not_found += 1
        out_lines.append(comp_str)
        i += 1
    else:
        out_lines.append(line)
        i += 1

print(f"Updated {updated} components, {not_found} not found in PL.")

with open(out_file, 'w') as f:
    f.writelines(out_lines)
print(f"Wrote to {out_file}")
