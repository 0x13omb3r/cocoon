#!/usr/bin/env python3
"""
Simple translation script.

Usage:
    echo "Hello world" | python simple_translate.py "Russian (ru)"
    python simple_translate.py "German (de)" --query "Hello world"
    python simple_translate.py "Chinese (zh)" --query-file input.txt
"""

import sys
import argparse
from translate import translate, add_translate_args, config_from_args


def main():
    parser = argparse.ArgumentParser(description='Translate text using LLM')
    parser.add_argument('target_lang', nargs='?', default='German (de)',
                        help='Target language (default: German (de))')
    parser.add_argument('--query', type=str,
                        help='Text to translate (alternative to stdin)')
    parser.add_argument('--query-file', type=str,
                        help='Read text from file (alternative to stdin)')
    add_translate_args(parser)
    
    args = parser.parse_args()
    
    if args.query and args.query_file:
        parser.error('Cannot specify both --query and --query-file')
    
    try:
        # Get text from query, query-file, or stdin
        if args.query:
            text = args.query
        elif args.query_file:
            with open(args.query_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        else:
            if sys.stdin.isatty():
                print('Paste text (Ctrl+D when done):', file=sys.stderr)
            text = sys.stdin.read().strip()

        if not text:
            sys.exit("No text provided.")

        # Load config from file or args (config_from_args handles both)
        config = config_from_args(args)

        print(f"Translating to {args.target_lang}...", file=sys.stderr)
        print(f"  Endpoint: {config.endpoint}" + (" (Azure)" if config.use_azure else ""), file=sys.stderr)
        print(f"  Format: {config.prompt_format}", file=sys.stderr)

        result = translate(text, args.target_lang, config)
        
        if args.verbose:
            if result.timing:
                print(f"  Time: {result.timing.duration:.3f}s", file=sys.stderr)
            if result.debug_data:
                print(f"\n--- Debug Info ---", file=sys.stderr)
                for key, value in result.debug_data.items():
                    print(f"  {key}: {value}", file=sys.stderr)
                print(f"--- End Debug ---\n", file=sys.stderr)
        
        print(result.translation)

    except KeyboardInterrupt:
        sys.exit("\nCancelled.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
