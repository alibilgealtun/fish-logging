"""Configuration package for fish logging system.

Provides a comprehensive configuration system with support for:
- Multiple configuration sources (defaults, files, environment, CLI)
- Hierarchical configuration with proper precedence
- Type-safe configuration objects with validation
- Simplified access through facade pattern

Main components:
- config.py: Core configuration dataclasses and loader
- service.py: Facade for simplified configuration access
"""

