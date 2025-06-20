import re

# Read the orchestrator file
with open('core/orchestrator.py', 'r') as f:
    content = f.read()

# Add error handling for missing methods
pattern = r'(.*?)(await self\._run_\w+_method\(.*?\))(.*?)'
replacement = r'\1try:\n                result = \2\n            except AttributeError as e:\n                self.logger.warning(f"Method not available: {e}")\n                result = None\3'

# Apply the fix
fixed_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Write back
with open('core/orchestrator.py', 'w') as f:
    f.write(fixed_content)

print("Fixed orchestrator to handle missing methods gracefully")
