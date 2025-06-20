#!/usr/bin/env python3
"""
Fix Tool Registration Issues in StegAnalyzer
Specifically addresses the "Tool file_analyzer not available for basic_analysis" warning
"""

import re
from pathlib import Path
from typing import Dict, Any

def main():
    """Fix tool registration and method mapping issues"""
    print("🔧 Fixing Tool Registration Issues")
    print("=" * 50)
    
    project_root = Path(".")
    
    # Step 1: Fix orchestrator tool mapping
    fix_orchestrator_tool_mapping(project_root)
    
    # Step 2: Fix missing file analyzer
    fix_file_analyzer_issue(project_root)
    
    # Step 3: Update tool initialization
    fix_tool_initialization(project_root)
    
    # Step 4: Fix method-to-tool mapping
    fix_method_tool_mapping(project_root)
    
    print("\n✅ All tool registration issues fixed!")
    print("\n🎯 The warnings should be eliminated:")
    print("   • file_analyzer tool properly registered")
    print("   • Method-to-tool mapping corrected")
    print("   • Tool initialization improved")
    
def fix_orchestrator_tool_mapping(project_root: Path):
    """Fix the orchestrator tool mapping"""
    print("\n🔧 Fixing orchestrator tool mapping...")
    
    orchestrator_file = project_root / "core" / "orchestrator.py"
    if not orchestrator_file.exists():
        print("   ❌ orchestrator.py not found!")
        return
    
    with open(orchestrator_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the method-to-tool mapping
    if "def _get_tool_for_method" in content:
        # Replace the entire method with a corrected version
        method_pattern = r'def _get_tool_for_method\(self, method: str\).*?(?=\n    def|\n\nclass|\Z)'
        
        new_method = '''def _get_tool_for_method(self, method: str) -> str:
        """Get the appropriate tool for a given method"""
        method_tool_map = {
            # File analysis methods
            "magic_analysis": "file_forensics",
            "basic_analysis": "file_forensics",  # Fix: map to file_forensics instead of file_analyzer
            "file_signature": "file_forensics",
            "entropy_analysis": "file_forensics",
            "hex_analysis": "file_forensics",
            
            # Image analysis methods
            "lsb_analysis": "classic_stego",
            "metadata_analysis": "metadata_carving",
            "image_forensics": "image_forensics",
            "steghide_extract": "classic_stego",
            "outguess_extract": "classic_stego",
            "zsteg_analysis": "classic_stego",
            
            # Audio analysis methods
            "audio_spectral": "audio_analysis",
            "audio_lsb": "audio_analysis",
            "audio_metadata": "metadata_carving",
            
            # Crypto analysis methods
            "crypto_analysis": "crypto_analysis",
            "hash_analysis": "crypto_analysis",
            "cipher_detection": "crypto_analysis",
            
            # ML analysis methods
            "ml_detection": "ml_detector",
            "anomaly_detection": "ml_detector",
            "statistical_analysis": "ml_detector",
            
            # LLM analysis methods
            "llm_analysis": "llm_analyzer",
            "pattern_recognition": "llm_analyzer",
            
            # Cascade analysis methods
            "cascade_analysis": "cascade_analyzer",
            "recursive_extract": "cascade_analyzer",
        }
        
        return method_tool_map.get(method, "file_forensics")  # Default to file_forensics'''
        
        content = re.sub(method_pattern, new_method, content, flags=re.DOTALL)
    
    # Also ensure the execute_method has proper error handling
    if "def execute_method" in content:
        # Add better error handling for missing tools
        execute_pattern = r'(def execute_method\(self, method.*?\n.*?)(\s+tool = self\.get_tool\(tool_name\).*?\n)'
        
        execute_replacement = r'\1\2' + '''        
        # Handle missing tools gracefully
        if not tool:
            self.logger.warning(f"Tool {tool_name} not available for {method}")
            return []
        
        if not hasattr(tool, 'execute_method'):
            self.logger.warning(f"Tool {tool_name} missing execute_method for {method}")
            return []
'''
        
        content = re.sub(execute_pattern, execute_replacement, content, flags=re.DOTALL)
    
    with open(orchestrator_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("   ✅ Orchestrator tool mapping fixed")

def fix_file_analyzer_issue(project_root: Path):
    """Fix the missing file_analyzer issue"""
    print("\n🗃️  Fixing file analyzer registration...")
    
    # Check if file_forensics tool exists and has the needed methods
    file_forensics_file = project_root / "tools" / "file_forensics.py"
    if not file_forensics_file.exists():
        print("   ❌ file_forensics.py not found!")
        return
    
    with open(file_forensics_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Ensure basic_analysis method exists
    if "def basic_analysis" not in content:
        # Add basic_analysis method
        basic_analysis_method = '''
    def basic_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Basic file analysis - wrapper for magic_analysis"""
        try:
            return self.magic_analysis(file_path)
        except Exception as e:
            self.logger.error(f"Basic analysis failed: {e}")
            return []
'''
        
        # Insert before the last method or at the end of class
        class_end_pattern = r'(\n    def [^_].*?\n        .*?\n\n)(\nclass|\Z)'
        if re.search(class_end_pattern, content, re.DOTALL):
            content = re.sub(
                class_end_pattern,
                r'\1' + basic_analysis_method + r'\2',
                content,
                count=1,
                flags=re.DOTALL
            )
        else:
            # Fallback: add before end of class
            content = content.replace('\nclass ', basic_analysis_method + '\n\nclass ')
    
    # Ensure execute_method properly handles basic_analysis
    if "def execute_method" in content:
        execute_pattern = r'(def execute_method\(self, method.*?\n.*?)(        return.*?\n)'
        
        execute_fix = '''        # Handle method name aliases
        if method == "basic_analysis":
            return self.basic_analysis(file_path)
        elif method == "file_signature":
            return self.magic_analysis(file_path)
        
'''
        
        content = re.sub(execute_pattern, r'\1' + execute_fix + r'\2', content, flags=re.DOTALL)
    
    with open(file_forensics_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("   ✅ File analyzer methods added to file_forensics")

def fix_tool_initialization(project_root: Path):
    """Fix tool initialization in orchestrator"""
    print("\n🔧 Fixing tool initialization...")
    
    orchestrator_file = project_root / "core" / "orchestrator.py"
    if not orchestrator_file.exists():
        return
    
    with open(orchestrator_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Ensure all tools are properly initialized
    if "def __init__" in content:
        init_pattern = r'(def __init__\(self, config.*?\n.*?)(        # Initialize.*?\n)'
        
        tool_init_code = '''        # Ensure all core tools are available
        self.tool_registry = {}
        
        # Register tools by name for easy lookup
        if self.file_tools:
            self.tool_registry['file_forensics'] = self.file_tools
            self.tool_registry['file_analyzer'] = self.file_tools  # Alias
        
        if self.classic_tools:
            self.tool_registry['classic_stego'] = self.classic_tools
        
        if self.image_tools:
            self.tool_registry['image_forensics'] = self.image_tools
        
        if self.audio_tools:
            self.tool_registry['audio_analysis'] = self.audio_tools
        
        if self.crypto_tools:
            self.tool_registry['crypto_analysis'] = self.crypto_tools
        
        if self.metadata_tools:
            self.tool_registry['metadata_carving'] = self.metadata_tools
        
        if self.ml_detector:
            self.tool_registry['ml_detector'] = self.ml_detector
        
        if self.llm_analyzer:
            self.tool_registry['llm_analyzer'] = self.llm_analyzer
        
        if self.cascade_analyzer:
            self.tool_registry['cascade_analyzer'] = self.cascade_analyzer
        
'''
        content = re.sub(init_pattern, r'\1' + tool_init_code + r'\2', content, flags=re.DOTALL)
    
    # Update get_tool method to use registry
    if "def get_tool" in content:
        get_tool_pattern = r'def get_tool\(self, tool_name: str\).*?(?=\n    def|\n\nclass|\Z)'
        
        new_get_tool = '''def get_tool(self, tool_name: str):
        """Get tool by name with fallback handling"""
        # Use registry first
        if hasattr(self, 'tool_registry') and tool_name in self.tool_registry:
            return self.tool_registry[tool_name]
        
        # Fallback to original logic
        tool_map = {
            'file_forensics': self.file_tools,
            'file_analyzer': self.file_tools,  # Alias for file_forensics
            'classic_stego': self.classic_tools,
            'image_forensics': self.image_tools,
            'audio_analysis': self.audio_tools,
            'crypto_analysis': self.crypto_tools,
            'metadata_carving': self.metadata_tools,
            'ml_detector': self.ml_detector,
            'llm_analyzer': self.llm_analyzer,
            'cascade_analyzer': self.cascade_analyzer
        }
        
        return tool_map.get(tool_name)'''
        
        content = re.sub(get_tool_pattern, new_get_tool, content, flags=re.DOTALL)
    
    with open(orchestrator_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("   ✅ Tool initialization improved")

def fix_method_tool_mapping(project_root: Path):
    """Fix method-to-tool mapping in task creation"""
    print("\n🎯 Fixing method-to-tool mapping...")
    
    orchestrator_file = project_root / "core" / "orchestrator.py"
    if not orchestrator_file.exists():
        return
    
    with open(orchestrator_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix _create_generic_analysis_tasks
    if "_create_generic_analysis_tasks" in content:
        # Ensure basic_analysis maps to file_forensics
        content = re.sub(
            r'tool_name="file_analyzer"',
            'tool_name="file_forensics"',
            content
        )
        
        # Fix any remaining basic_analysis mappings
        basic_analysis_pattern = r'(AnalysisTask.*?method="basic_analysis".*?tool_name=")([^"]*)"'
        content = re.sub(
            basic_analysis_pattern,
            r'\1file_forensics"',
            content
        )
    
    # Update any hardcoded method mappings
    method_mappings = {
        'file_analyzer': 'file_forensics',
        'basic_file_analysis': 'file_forensics',
        'file_signature': 'file_forensics'
    }
    
    for old_name, new_name in method_mappings.items():
        content = content.replace(f'"{old_name}"', f'"{new_name}"')
        content = content.replace(f"'{old_name}'", f"'{new_name}'")
    
    with open(orchestrator_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("   ✅ Method-to-tool mapping corrected")

if __name__ == "__main__":
    main()
