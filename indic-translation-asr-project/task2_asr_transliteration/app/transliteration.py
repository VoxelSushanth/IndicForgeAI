"""
Transliteration Engine for Tamil Script Conversion.

This module provides a comprehensive transliteration engine supporting
multiple romanization schemes and Indian scripts using the indic-transliteration
library. It handles edge cases gracefully and preserves non-Tamil text.
"""

import logging
from typing import List, Optional
from pathlib import Path

# Configure module logger
logger = logging.getLogger(__name__)


class TransliterationEngine:
    """
    Multi-scheme transliteration engine for Tamil text.
    
    This class provides bidirectional transliteration between Tamil script
    and various romanization schemes (ITRANS, HK, IAST, SLP1, WX) as well
    as other Indian scripts (Devanagari). It handles mixed-script input
    and preserves non-Indic Unicode characters.
    
    Supported schemes:
        - ITRANS: Common romanization for Indic languages
        - HK: Harvard-Kyoto scheme (ASCII-only)
        - IAST: International Alphabet of Sanskrit Transliteration
        - SLP1: Sanskrit Library Phonetic Basic
        - WX: IIT Bombay WX scheme
        - Devanagari: Devanagari script conversion
        - Tamil: Tamil script (target or source)
    
    Attributes:
        source_scheme: Source script/scheme name
        target_scheme: Target script/scheme name
    
    Example:
        >>> engine = TransliterationEngine(source_scheme="Tamil", target_scheme="ITRANS")
        >>> engine.transliterate("வணக்கம்")
        'vaNakkam'
    """
    
    # Valid transliteration schemes
    VALID_SCHEMES = {"ITRANS", "HK", "IAST", "SLP1", "WX", "Devanagari", "Tamil"}
    
    # Unicode range for Tamil script: U+0B80 to U+0BFF
    TAMIL_UNICODE_START = 0x0B80
    TAMIL_UNICODE_END = 0x0BFF
    
    def __init__(self, source_scheme: str = "Tamil", target_scheme: str = "ITRANS") -> None:
        """
        Initialize the transliteration engine.
        
        Args:
            source_scheme: Source script/scheme (default: "Tamil")
            target_scheme: Target script/scheme (default: "ITRANS")
        
        Raises:
            ValueError: If either scheme is not in the valid schemes list
        """
        # Validate schemes
        source_upper = source_scheme.upper() if source_scheme else ""
        target_upper = target_scheme.upper() if target_scheme else ""
        
        # Handle case-insensitive matching
        valid_schemes_upper = {s.upper(): s for s in self.VALID_SCHEMES}
        
        if source_upper not in valid_schemes_upper:
            raise ValueError(
                f"Invalid source scheme: {source_scheme}. "
                f"Valid schemes: {', '.join(self.VALID_SCHEMES)}"
            )
        if target_upper not in valid_schemes_upper:
            raise ValueError(
                f"Invalid target scheme: {target_scheme}. "
                f"Valid schemes: {', '.join(self.VALID_SCHEMES)}"
            )
        
        self.source_scheme = valid_schemes_upper[source_upper]
        self.target_scheme = valid_schemes_upper[target_upper]
        
        logger.info(
            f"TransliterationEngine initialized: {self.source_scheme} -> {self.target_scheme}"
        )
    
    def detect_script(self, text: str) -> str:
        """
        Detect the primary script of the input text.
        
        This method analyzes the Unicode codepoints in the text to determine
        whether it is primarily Tamil script, Latin script, or mixed content.
        
        Args:
            text: Input text to analyze
        
        Returns:
            Script identifier: "Tamil", "Latin", or "Mixed"
        """
        if not text:
            return "Empty"
        
        tamil_count = 0
        latin_count = 0
        other_count = 0
        
        for char in text:
            codepoint = ord(char)
            
            # Check if Tamil (U+0B80 to U+0BFF)
            if self.TAMIL_UNICODE_START <= codepoint <= self.TAMIL_UNICODE_END:
                tamil_count += 1
            # Check if Latin (basic ASCII letters)
            elif ('a' <= char <= 'z') or ('A' <= char <= 'Z'):
                latin_count += 1
            else:
                other_count += 1
        
        total = len(text)
        
        # Determine dominant script
        if total == 0:
            return "Empty"
        
        tamil_ratio = tamil_count / total
        latin_ratio = latin_count / total
        
        # If more than 50% Tamil characters
        if tamil_ratio > 0.5:
            return "Tamil"
        # If more than 50% Latin characters
        elif latin_ratio > 0.5:
            return "Latin"
        # Otherwise mixed
        else:
            return "Mixed"
    
    def transliterate(self, text: str) -> str:
        """
        Transliterate text from source scheme to target scheme.
        
        This method performs the actual transliteration using the
        indic_transliteration library. It handles empty strings,
        non-Tamil Unicode, and whitespace normalization gracefully.
        
        Args:
            text: Input text to transliterate
        
        Returns:
            Transliterated text with normalized whitespace
        
        Note:
            - Empty string input returns empty string
            - Non-Tamil Unicode characters are passed through unchanged
            - Leading/trailing whitespace is stripped
            - Internal whitespace is normalized (multiple spaces -> single space)
        """
        # Handle empty string
        if not text:
            logger.debug("Empty text provided, returning empty string")
            return ""
        
        try:
            from indic_transliteration import sanscript
            
            # Strip and normalize whitespace
            text = ' '.join(text.split())
            
            # Detect script for logging
            detected = self.detect_script(text)
            logger.debug(f"Detected script: {detected}")
            
            # Perform transliteration
            result = sanscript.transliterate(
                text,
                from_scheme=self.source_scheme,
                to_scheme=self.target_scheme
            )
            
            # Normalize output whitespace
            result = ' '.join(result.split())
            
            logger.debug(
                f"Transliterated: '{text[:30]}...' -> '{result[:30]}...'"
            )
            
            return result
            
        except ImportError:
            logger.error("indic_transliteration not installed")
            raise RuntimeError(
                "indic_transliteration library required. "
                "Install with: pip install indic-transliteration"
            )
        except Exception as e:
            logger.warning(f"Transliteration failed for text '{text[:30]}...': {e}")
            # Return original text on failure (passthrough)
            return text
    
    def batch_transliterate(self, texts: List[str]) -> List[str]:
        """
        Transliterate a list of texts, preserving order and handling failures.
        
        This method processes multiple texts sequentially, ensuring that
        individual failures don't affect the entire batch. Failed items
        return their original text.
        
        Args:
            texts: List of strings to transliterate
        
        Returns:
            List of transliterated strings in the same order as input
        
        Note:
            - Order is preserved exactly
            - Individual failures return original text (no exceptions raised)
            - Empty strings in input produce empty strings in output
        """
        if not texts:
            return []
        
        results: List[str] = []
        
        for i, text in enumerate(texts):
            try:
                result = self.transliterate(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch item {i} failed: {e}")
                # Preserve original on failure
                results.append(text if text else "")
        
        logger.info(f"Batch transliteration complete: {len(results)} items")
        return results
    
    def set_schemes(self, source_scheme: str, target_scheme: str) -> None:
        """
        Update the source and target transliteration schemes.
        
        Args:
            source_scheme: New source script/scheme
            target_scheme: New target script/scheme
        
        Raises:
            ValueError: If either scheme is invalid
        """
        # Validate new schemes
        source_upper = source_scheme.upper() if source_scheme else ""
        target_upper = target_scheme.upper() if target_scheme else ""
        
        valid_schemes_upper = {s.upper(): s for s in self.VALID_SCHEMES}
        
        if source_upper not in valid_schemes_upper:
            raise ValueError(
                f"Invalid source scheme: {source_scheme}. "
                f"Valid schemes: {', '.join(self.VALID_SCHEMES)}"
            )
        if target_upper not in valid_schemes_upper:
            raise ValueError(
                f"Invalid target scheme: {target_scheme}. "
                f"Valid schemes: {', '.join(self.VALID_SCHEMES)}"
            )
        
        old_source = self.source_scheme
        old_target = self.target_scheme
        
        self.source_scheme = valid_schemes_upper[source_upper]
        self.target_scheme = valid_schemes_upper[target_upper]
        
        logger.info(
            f"Scheme changed: {old_source}->{old_target} -> "
            f"{self.source_scheme}->{self.target_scheme}"
        )
