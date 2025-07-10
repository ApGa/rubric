import json
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AnimationEffect:
    """Represents a single animation effect"""

    slide_id: str
    element_id: str
    effect_type: str
    trigger: str
    delay: float
    duration: float
    order: int
    # Note, element_text may sometimes be a superset of the finer grained text actually animated.
    element_text: Optional[str] = None
    element_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slide_id": self.slide_id,
            "element_id": self.element_id,
            "effect_type": self.effect_type,
            "trigger": self.trigger,
            "delay": self.delay,
            "duration": self.duration,
            "order": self.order,
            "element_text": self.element_text,
            "element_type": self.element_type,
        }


@dataclass
class SlideTransition:
    """Represents a slide transition"""

    slide_id: str
    transition_type: str
    duration: float
    direction: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slide_id": self.slide_id,
            "transition_type": self.transition_type,
            "duration": self.duration,
            "direction": self.direction,
        }


@dataclass
class Slide:
    """Represents a slide with its metadata"""

    slide_id: str
    slide_number: int
    title: Optional[str] = None
    layout_type: Optional[str] = None
    element_count: int = 0
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slide_id": self.slide_id,
            "slide_number": self.slide_number,
            "title": self.title,
            "layout_type": self.layout_type,
            "element_count": self.element_count,
            "notes": self.notes,
        }


@dataclass
class PowerPointDiff:
    """Container for PowerPoint differences"""

    added_animations: List[AnimationEffect]
    removed_animations: List[AnimationEffect]
    modified_animations: List[Tuple[AnimationEffect, AnimationEffect]]
    added_transitions: List[SlideTransition]
    removed_transitions: List[SlideTransition]
    modified_transitions: List[Tuple[SlideTransition, SlideTransition]]
    added_slides: List[Slide]
    removed_slides: List[Slide]
    modified_slides: List[Tuple[Slide, Slide]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "added_animations": [anim.to_dict() for anim in self.added_animations],
            "removed_animations": [anim.to_dict() for anim in self.removed_animations],
            "modified_animations": [
                [old.to_dict(), new.to_dict()] for old, new in self.modified_animations
            ],
            "added_transitions": [trans.to_dict() for trans in self.added_transitions],
            "removed_transitions": [trans.to_dict() for trans in self.removed_transitions],
            "modified_transitions": [
                [old.to_dict(), new.to_dict()] for old, new in self.modified_transitions
            ],
            "added_slides": [slide.to_dict() for slide in self.added_slides],
            "removed_slides": [slide.to_dict() for slide in self.removed_slides],
            "modified_slides": [
                [old.to_dict(), new.to_dict()] for old, new in self.modified_slides
            ],
        }


class PowerPointDiffEvaluator:
    """Evaluates differences between PowerPoint files focusing on animations and transitions"""

    def __init__(self):
        self.namespaces = {
            "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
            "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
            "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        }
        self.slide_elements_cache = {}  # Cache for slide elements

    def extract_animations_and_transitions(
        self, pptx_path: str
    ) -> Tuple[List[AnimationEffect], List[SlideTransition]]:
        """Extract animations and transitions from a PowerPoint file"""
        animations = []
        transitions = []

        try:
            with zipfile.ZipFile(pptx_path, "r") as pptx:
                # Get slide files
                slide_files = [
                    f
                    for f in pptx.namelist()
                    if f.startswith("ppt/slides/slide") and f.endswith(".xml")
                ]

                for slide_file in slide_files:
                    slide_id = self._extract_slide_id(slide_file)

                    # Read slide XML
                    slide_xml = pptx.read(slide_file)
                    slide_root = ET.fromstring(slide_xml)

                    # Cache slide elements for text lookup
                    self.slide_elements_cache[slide_id] = self._build_element_map(slide_root)

                    # Extract transitions
                    transition = self._extract_transition(slide_root, slide_id)
                    if transition:
                        transitions.append(transition)

                    # Extract animations
                    slide_animations = self._extract_animations(slide_root, slide_id)
                    animations.extend(slide_animations)

        except Exception as e:
            print(f"Error processing {pptx_path}: {e}")

        return animations, transitions

    def extract_slides(self, pptx_path: str) -> List[Slide]:
        """Extract slide metadata from a PowerPoint file"""
        slides = []

        try:
            with zipfile.ZipFile(pptx_path, "r") as pptx:
                # Get slide files with more robust pattern matching
                slide_files = []
                for file_path in pptx.namelist():
                    # Look for slide files in the standard location
                    if (
                        file_path.startswith("ppt/slides/slide")
                        and file_path.endswith(".xml")
                        and file_path.count("/") == 2
                    ):  # Ensure it's not in a subdirectory
                        slide_files.append(file_path)

                if not slide_files:
                    print(f"Warning: No slide files found in {pptx_path}")
                    return slides

                # Sort by numeric slide number, not lexicographic
                def extract_slide_number(slide_file: str) -> int:
                    try:
                        # Extract number from "ppt/slides/slideN.xml"
                        filename = slide_file.split("/")[-1]  # get "slideN.xml"
                        number_str = filename.replace("slide", "").replace(".xml", "")  # get "N"
                        return int(number_str)
                    except (ValueError, IndexError):
                        # Fallback: if we can't parse the number, use 999999 to put it at the end
                        print(f"Warning: Could not parse slide number from {slide_file}")
                        return 999999

                slide_files.sort(key=extract_slide_number)

                # Extract metadata for each slide
                for i, slide_file in enumerate(slide_files):
                    try:
                        slide_id = self._extract_slide_id(slide_file)

                        # Use the extracted number if available, otherwise use sequence
                        original_number = extract_slide_number(slide_file)
                        slide_number = original_number if original_number != 999999 else i + 1

                        # Read slide XML
                        slide_xml = pptx.read(slide_file)
                        slide_root = ET.fromstring(slide_xml)

                        # Extract slide metadata
                        slide = self._extract_slide_metadata(
                            slide_root, slide_id, slide_number, pptx
                        )
                        slides.append(slide)

                    except Exception as e:
                        print(f"Warning: Error processing slide {slide_file}: {e}")
                        continue  # Skip this slide but continue with others

        except Exception as e:
            print(f"Error processing slides from {pptx_path}: {e}")

        return slides

    def _extract_slide_metadata(
        self, slide_root: ET.Element, slide_id: str, slide_number: int, pptx: zipfile.ZipFile
    ) -> Slide:
        """Extract metadata from a single slide"""
        # Extract title
        title = self._extract_slide_title(slide_root)

        # Extract layout type
        layout_type = self._extract_layout_type(slide_root)

        # Count elements
        element_count = self._count_slide_elements(slide_root)

        # Extract notes (if available)
        notes = self._extract_slide_notes(slide_id, pptx)

        return Slide(
            slide_id=slide_id,
            slide_number=slide_number,
            title=title,
            layout_type=layout_type,
            element_count=element_count,
            notes=notes,
        )

    def _extract_slide_title(self, slide_root: ET.Element) -> Optional[str]:
        """Extract title from slide"""
        # Look for title placeholder
        title_elements = slide_root.findall(".//p:sp", self.namespaces)

        for shape in title_elements:
            # Check if this is a title placeholder
            nv_sp_pr = shape.find(".//p:nvSpPr", self.namespaces)
            if nv_sp_pr is not None:
                ph = nv_sp_pr.find(".//p:ph", self.namespaces)
                if ph is not None:
                    ph_type = ph.get("type")
                    if ph_type in ["title", "ctrTitle"]:
                        # Extract text from this shape
                        return self._extract_text_from_shape(shape)

                # Also check by name
                c_nv_pr = nv_sp_pr.find(".//p:cNvPr", self.namespaces)
                if c_nv_pr is not None:
                    name = c_nv_pr.get("name", "").lower()
                    if "title" in name:
                        return self._extract_text_from_shape(shape)

        return None

    def _extract_layout_type(self, slide_root: ET.Element) -> Optional[str]:
        """Extract layout type from slide"""
        # Look for layout reference - this is more complex in practice
        # For now, we'll determine based on placeholders present
        placeholders = slide_root.findall(".//p:ph", self.namespaces)

        ph_types = set()
        for ph in placeholders:
            ph_type = ph.get("type")
            if ph_type:
                ph_types.add(ph_type)

        # Determine layout based on placeholder types
        if "title" in ph_types and "body" in ph_types:
            return "Title and Content"
        elif "title" in ph_types and not ph_types - {"title"}:
            return "Title Only"
        elif "ctrTitle" in ph_types and "subTitle" in ph_types:
            return "Title Slide"
        elif "body" in ph_types and len(ph_types) > 1:
            return "Content with Title"
        elif not ph_types:
            return "Blank"
        else:
            return "Custom Layout"

    def _count_slide_elements(self, slide_root: ET.Element) -> int:
        """Count the number of elements on a slide"""
        # Count shapes
        shapes = slide_root.findall(".//p:sp", self.namespaces)
        return len(shapes)

    def _extract_slide_notes(self, slide_id: str, pptx: zipfile.ZipFile) -> Optional[str]:
        """Extract notes for a slide"""
        try:
            # Extract slide number from slide_id (e.g., 'slide1' -> '1')
            slide_num = slide_id.replace("slide", "")
            notes_file = f"ppt/notesSlides/notesSlide{slide_num}.xml"

            if notes_file in pptx.namelist():
                notes_xml = pptx.read(notes_file)
                notes_root = ET.fromstring(notes_xml)

                # Extract text from notes
                text_elements = notes_root.findall(".//a:t", self.namespaces)
                notes_text = []
                for text_elem in text_elements:
                    if text_elem.text:
                        notes_text.append(text_elem.text)

                full_notes = " ".join(notes_text).strip()
                return full_notes if full_notes else None

        except Exception:
            pass  # Notes are optional, don't fail if we can't read them

        return None

    def _build_element_map(self, slide_root: ET.Element) -> Dict[str, Dict[str, str]]:
        """Build a map of element IDs to their text content and types"""
        element_map = {}

        # Find all shapes in the slide
        shapes = slide_root.findall(".//p:sp", self.namespaces)

        for shape in shapes:
            # Get shape ID
            nv_sp_pr = shape.find(".//p:nvSpPr", self.namespaces)
            if nv_sp_pr is not None:
                c_nv_pr = nv_sp_pr.find(".//p:cNvPr", self.namespaces)
                if c_nv_pr is not None:
                    shape_id = c_nv_pr.get("id")
                    shape_name = c_nv_pr.get("name", "")

                    if shape_id:
                        # Extract text content
                        text_content = self._extract_text_from_shape(shape)

                        # Determine element type
                        element_type = self._determine_element_type(shape, shape_name)

                        element_map[shape_id] = {
                            "text": text_content,
                            "type": element_type,
                            "name": shape_name,
                        }

        # Also check for group shapes
        groups = slide_root.findall(".//p:grpSp", self.namespaces)
        for group in groups:
            group_shapes = group.findall(".//p:sp", self.namespaces)
            for shape in group_shapes:
                nv_sp_pr = shape.find(".//p:nvSpPr", self.namespaces)
                if nv_sp_pr is not None:
                    c_nv_pr = nv_sp_pr.find(".//p:cNvPr", self.namespaces)
                    if c_nv_pr is not None:
                        shape_id = c_nv_pr.get("id")
                        shape_name = c_nv_pr.get("name", "")

                        if shape_id:
                            text_content = self._extract_text_from_shape(shape)
                            element_type = self._determine_element_type(shape, shape_name)

                            element_map[shape_id] = {
                                "text": text_content,
                                "type": element_type,
                                "name": shape_name,
                            }

        return element_map

    def _extract_text_from_shape(self, shape: ET.Element) -> Optional[str]:
        """Extract all text content from a shape"""
        text_parts = []

        # Find all text elements
        text_elements = shape.findall(".//a:t", self.namespaces)
        for text_elem in text_elements:
            if text_elem.text:
                text_parts.append(text_elem.text)

        # Join all text parts
        full_text = "".join(text_parts).strip()

        # Clean up whitespace
        full_text = re.sub(r"\s+", " ", full_text)

        return full_text if full_text else None

    def _determine_element_type(self, shape: ET.Element, shape_name: str) -> str:
        """Determine the type of element based on shape properties"""
        # Check if it's a text box
        if "TextBox" in shape_name or "Text Box" in shape_name:
            return "textbox"

        # Check if it's a title
        if "Title" in shape_name:
            return "title"

        # Check if it's a content placeholder
        if "Content Placeholder" in shape_name or "Content" in shape_name:
            return "content"

        # Check if it has text body
        tx_body = shape.find(".//p:txBody", self.namespaces)
        if tx_body is not None:
            return "text_shape"

        # Check for specific shape types
        sp_pr = shape.find(".//p:spPr", self.namespaces)
        if sp_pr is not None:
            # Check for preset geometry
            prst_geom = sp_pr.find(".//a:prstGeom", self.namespaces)
            if prst_geom is not None:
                prst = prst_geom.get("prst", "")
                if prst == "rect":
                    return "rectangle"
                elif prst == "ellipse":
                    return "ellipse"
                elif prst in ["star5", "star6", "star8"]:
                    return "star"
                elif prst in ["rightArrow", "leftArrow", "upArrow", "downArrow"]:
                    return "arrow"
                else:
                    return f"shape_{prst}"

        return "shape"

    def _extract_slide_id(self, slide_file: str) -> str:
        """Extract slide ID from file path"""
        return slide_file.split("/")[-1].replace(".xml", "")

    def _extract_transition(
        self, slide_root: ET.Element, slide_id: str
    ) -> Optional[SlideTransition]:
        """Extract transition information from slide XML"""
        transition_elem = slide_root.find(".//p:transition", self.namespaces)
        if transition_elem is None:
            return None

        # Get transition type
        transition_type = "none"
        for child in transition_elem:
            if "}" in child.tag:
                # Extract the tag name without the namespace, e.g., 'fade' from '{...}fade'
                tag_name = child.tag.split("}")[-1]

                # Child elements of <p:transition> can be the transition type (e.g., <p:fade>)
                # or other properties (e.g., <p:soundAc> for sound).
                # We assume that any child that is not a known property is the transition type.
                if tag_name != "soundAc":
                    transition_type = tag_name
                    break  # Assume the first such element is the transition type

            # if child.tag.endswith('}fade'):
            #     transition_type = "fade"
            # elif child.tag.endswith('}push'):
            #     transition_type = "push"
            # elif child.tag.endswith('}wipe'):
            #     transition_type = "wipe"
            # elif child.tag.endswith('}cut'):
            #     transition_type = "cut"
            # elif child.tag.endswith('}dissolve'):
            #     transition_type = "dissolve"
            # elif child.tag.endswith('}morph'):
            #     transition_type = "morph"
            # # Add more transition types as needed

        # Get duration (in milliseconds, convert to seconds)
        duration_attr = transition_elem.get("dur", "500")
        duration = float(duration_attr) / 1000.0

        # Get direction if available
        direction = None
        for child in transition_elem:
            direction_attr = child.get("dir")
            if direction_attr:
                direction = direction_attr
                break

        return SlideTransition(
            slide_id=slide_id,
            transition_type=transition_type,
            duration=duration,
            direction=direction,
        )

    def _extract_animations(self, slide_root: ET.Element, slide_id: str) -> List[AnimationEffect]:
        """Extract animation effects from slide XML"""
        animations = []

        # Find timing root
        timing_root = slide_root.find(".//p:timing", self.namespaces)
        if timing_root is None:
            return animations

        parent_map = {c: p for p in timing_root.iter() for c in p}

        # Look for various animation elements
        animation_elements = []

        # Find animation effects
        animation_elements.extend(timing_root.findall(".//p:animEffect", self.namespaces))

        # Find animation motions
        animation_elements.extend(timing_root.findall(".//p:animMotion", self.namespaces))

        # Find animation colors
        animation_elements.extend(timing_root.findall(".//p:animClr", self.namespaces))

        # Find animation rotations
        animation_elements.extend(timing_root.findall(".//p:animRot", self.namespaces))

        # Find animation scales
        animation_elements.extend(timing_root.findall(".//p:animScale", self.namespaces))

        # Find set animations (appear/disappear)
        animation_elements.extend(timing_root.findall(".//p:set", self.namespaces))

        # Find animation groups/sequences
        par_elements = timing_root.findall(".//p:par", self.namespaces)
        for par in par_elements:
            # Look for animations within parallel groups
            animation_elements.extend(par.findall(".//p:animEffect", self.namespaces))
            animation_elements.extend(par.findall(".//p:animMotion", self.namespaces))
            animation_elements.extend(par.findall(".//p:animClr", self.namespaces))
            animation_elements.extend(par.findall(".//p:animRot", self.namespaces))
            animation_elements.extend(par.findall(".//p:animScale", self.namespaces))
            animation_elements.extend(par.findall(".//p:set", self.namespaces))

        # Parse each animation element
        for i, anim_elem in enumerate(animation_elements):
            animation = self._parse_animation_effect(anim_elem, slide_id, i, parent_map)
            if animation:
                animations.append(animation)

        return animations

    def _parse_animation_effect(
        self,
        effect_elem: ET.Element,
        slide_id: str,
        order: int,
        parent_map: Dict[ET.Element, ET.Element],
    ) -> Optional[AnimationEffect]:
        """Parse a single animation effect element"""
        try:
            # Determine animation type based on element tag
            tag_name = effect_elem.tag.split("}")[-1] if "}" in effect_elem.tag else effect_elem.tag

            # effect_type = "unknown"
            # if tag_name == "animEffect":
            #     effect_type = effect_elem.get('filter', 'effect')
            # elif tag_name == "animMotion":
            #     effect_type = "motion"
            # elif tag_name == "animClr":
            #     effect_type = "color"
            # elif tag_name == "animRot":
            #     effect_type = "rotation"
            # elif tag_name == "animScale":
            #     effect_type = "scale"
            # elif tag_name == "set":
            #     effect_type = "set"

            effect_type = tag_name

            # Get target element ID - try multiple approaches
            element_id = "unknown"

            # First try to find target element directly
            target_elem = effect_elem.find(".//p:tgtEl", self.namespaces)
            if target_elem is not None:
                # Try shape target
                sp_tgt = target_elem.find(".//p:spTgt", self.namespaces)
                if sp_tgt is not None:
                    element_id = sp_tgt.get("spid", "unknown")
                else:
                    # Try ink target
                    ink_tgt = target_elem.find(".//p:inkTgt", self.namespaces)
                    if ink_tgt is not None:
                        element_id = ink_tgt.get("spid", "unknown")
                    else:
                        # Try text target
                        txt_tgt = target_elem.find(".//p:txtTgt", self.namespaces)
                        if txt_tgt is not None:
                            element_id = txt_tgt.get("spid", "unknown")

            # Try to find target in parent container if not found directly
            if element_id == "unknown":
                parent = parent_map.get(effect_elem)
                while parent is not None and element_id == "unknown":
                    target_elem = parent.find(".//p:tgtEl", self.namespaces)
                    if target_elem is not None:
                        sp_tgt = target_elem.find(".//p:spTgt", self.namespaces)
                        if sp_tgt is not None:
                            element_id = sp_tgt.get("spid", "unknown")
                            break
                    parent = parent_map.get(parent)

            # Get timing information
            timing_elem = effect_elem.find(".//p:cTn", self.namespaces)
            delay = 0.0
            duration = 1.0
            if timing_elem is not None:
                delay_str = timing_elem.get("delay", "0")
                dur_str = timing_elem.get("dur", "1000")

                # Handle special duration values
                if dur_str == "indefinite":
                    duration = 0.0
                else:
                    try:
                        delay = float(delay_str) / 1000.0
                        duration = float(dur_str) / 1000.0
                    except ValueError:
                        delay = 0.0
                        duration = 1.0

            # Get trigger type - look in various places
            trigger = "onClick"  # Default

            # Look for condition elements
            cond_elem = effect_elem.find(".//p:cond", self.namespaces)
            if cond_elem is not None:
                trigger = cond_elem.get("evt", "onClick")
            else:
                # Look in parent elements
                parent = parent_map.get(effect_elem)
                while parent is not None and trigger == "onClick":
                    cond_elem = parent.find(".//p:cond", self.namespaces)
                    if cond_elem is not None:
                        trigger = cond_elem.get("evt", "onClick")
                        break
                    parent = parent_map.get(parent)

            # Map common trigger types
            trigger_map = {
                "onBegin": "withPrevious",
                "onEnd": "afterPrevious",
                "onClick": "onClick",
                "onDblClick": "onDoubleClick",
                "onMouseOver": "onMouseOver",
                "onMouseOut": "onMouseOut",
            }
            trigger = trigger_map.get(trigger, trigger)

            # Get text content and element type from cache
            element_text = None
            element_type = None
            if (
                slide_id in self.slide_elements_cache
                and element_id in self.slide_elements_cache[slide_id]
            ):
                element_info = self.slide_elements_cache[slide_id][element_id]
                element_text = element_info["text"]
                element_type = element_info["type"]

            return AnimationEffect(
                slide_id=slide_id,
                element_id=element_id,
                effect_type=effect_type,
                trigger=trigger,
                delay=delay,
                duration=duration,
                order=order,
                element_text=element_text,
                element_type=element_type,
            )

        except Exception as e:
            print(f"Error parsing animation effect: {e}")
            return None

    def compare_files(self, before_path: str, after_path: str) -> PowerPointDiff:
        """Compare two PowerPoint files and return differences"""
        # Extract animations and transitions from both files
        before_animations, before_transitions = self.extract_animations_and_transitions(before_path)
        after_animations, after_transitions = self.extract_animations_and_transitions(after_path)

        # Extract slides from both files
        before_slides = self.extract_slides(before_path)
        after_slides = self.extract_slides(after_path)

        # Compare animations
        added_animations, removed_animations, modified_animations = self._compare_animations(
            before_animations, after_animations
        )

        # Compare transitions
        added_transitions, removed_transitions, modified_transitions = self._compare_transitions(
            before_transitions, after_transitions
        )

        # Compare slides
        added_slides, removed_slides, modified_slides = self._compare_slides(
            before_slides, after_slides
        )

        return PowerPointDiff(
            added_animations=added_animations,
            removed_animations=removed_animations,
            modified_animations=modified_animations,
            added_transitions=added_transitions,
            removed_transitions=removed_transitions,
            modified_transitions=modified_transitions,
            added_slides=added_slides,
            removed_slides=removed_slides,
            modified_slides=modified_slides,
        )

    def _compare_animations(
        self, before: List[AnimationEffect], after: List[AnimationEffect]
    ) -> Tuple[
        List[AnimationEffect], List[AnimationEffect], List[Tuple[AnimationEffect, AnimationEffect]]
    ]:
        """Compare animation lists"""
        # Create lookup dictionaries
        before_dict = {self._animation_key(anim): anim for anim in before}
        after_dict = {self._animation_key(anim): anim for anim in after}

        # Find differences
        added = [anim for key, anim in after_dict.items() if key not in before_dict]
        removed = [anim for key, anim in before_dict.items() if key not in after_dict]

        # Find modified animations
        modified = []
        for key in before_dict:
            if key in after_dict:
                before_anim = before_dict[key]
                after_anim = after_dict[key]
                if not self._animations_equal(before_anim, after_anim):
                    modified.append((before_anim, after_anim))

        return added, removed, modified

    def _compare_transitions(
        self, before: List[SlideTransition], after: List[SlideTransition]
    ) -> Tuple[
        List[SlideTransition], List[SlideTransition], List[Tuple[SlideTransition, SlideTransition]]
    ]:
        """Compare transition lists"""
        # Create lookup dictionaries
        before_dict = {trans.slide_id: trans for trans in before}
        after_dict = {trans.slide_id: trans for trans in after}

        # Find differences
        added = [trans for slide_id, trans in after_dict.items() if slide_id not in before_dict]
        removed = [trans for slide_id, trans in before_dict.items() if slide_id not in after_dict]

        # Find modified transitions
        modified = []
        for slide_id in before_dict:
            if slide_id in after_dict:
                before_trans = before_dict[slide_id]
                after_trans = after_dict[slide_id]
                if not self._transitions_equal(before_trans, after_trans):
                    modified.append((before_trans, after_trans))

        return added, removed, modified

    def _compare_slides(
        self, before: List[Slide], after: List[Slide]
    ) -> Tuple[List[Slide], List[Slide], List[Tuple[Slide, Slide]]]:
        """Compare slide metadata lists"""
        # Create lookup dictionaries
        before_dict = {slide.slide_id: slide for slide in before}
        after_dict = {slide.slide_id: slide for slide in after}

        # Find differences
        added = [slide for slide_id, slide in after_dict.items() if slide_id not in before_dict]
        removed = [slide for slide_id, slide in before_dict.items() if slide_id not in after_dict]

        # Find modified slides
        modified = []
        for slide_id in before_dict:
            if slide_id in after_dict:
                before_slide = before_dict[slide_id]
                after_slide = after_dict[slide_id]
                if not self._slides_equal(before_slide, after_slide):
                    modified.append((before_slide, after_slide))

        return added, removed, modified

    def _slides_equal(self, slide1: Slide, slide2: Slide) -> bool:
        """Check if two slides are equal"""
        return (
            slide1.title == slide2.title
            and slide1.layout_type == slide2.layout_type
            and slide1.element_count == slide2.element_count
            and slide1.notes == slide2.notes
        )

    def _animation_key(self, animation: AnimationEffect) -> str:
        """Generate a unique key for an animation"""
        return f"{animation.slide_id}_{animation.element_id}_{animation.order}"

    def _animations_equal(self, anim1: AnimationEffect, anim2: AnimationEffect) -> bool:
        """Check if two animations are equal"""
        return (
            anim1.effect_type == anim2.effect_type
            and anim1.trigger == anim2.trigger
            and abs(anim1.delay - anim2.delay) < 0.01
            and abs(anim1.duration - anim2.duration) < 0.01
        )

    def _transitions_equal(self, trans1: SlideTransition, trans2: SlideTransition) -> bool:
        """Check if two transitions are equal"""
        return (
            trans1.transition_type == trans2.transition_type
            and abs(trans1.duration - trans2.duration) < 0.01
            and trans1.direction == trans2.direction
        )

    def generate_concise_diff_report(self, diff: PowerPointDiff) -> str:
        """Generate a concise, human-readable diff report with text content"""
        report_parts = []

        # Added slides
        if diff.added_slides:
            report_parts.append(f"Added {len(diff.added_slides)} slides:")
            for slide in diff.added_slides:
                title_info = f" ('{slide.title}')" if slide.title else ""
                layout_info = f" [{slide.layout_type}]" if slide.layout_type else ""
                report_parts.append(
                    f"  - Slide {slide.slide_number}: {slide.slide_id}{layout_info}{title_info}"
                )

        # Removed slides
        if diff.removed_slides:
            report_parts.append(f"Removed {len(diff.removed_slides)} slides:")
            for slide in diff.removed_slides:
                title_info = f" ('{slide.title}')" if slide.title else ""
                layout_info = f" [{slide.layout_type}]" if slide.layout_type else ""
                report_parts.append(
                    f"  - Slide {slide.slide_number}: {slide.slide_id}{layout_info}{title_info}"
                )

        # Modified slides
        if diff.modified_slides:
            report_parts.append(f"Modified {len(diff.modified_slides)} slides:")
            for old, new in diff.modified_slides:
                changes = []
                if old.title != new.title:
                    changes.append(f"title: '{old.title}' → '{new.title}'")
                if old.layout_type != new.layout_type:
                    changes.append(f"layout: {old.layout_type} → {new.layout_type}")
                if old.element_count != new.element_count:
                    changes.append(f"elements: {old.element_count} → {new.element_count}")
                if old.notes != new.notes:
                    changes.append("notes changed")

                change_summary = ", ".join(changes) if changes else "unknown changes"
                report_parts.append(
                    f"  - Slide {new.slide_number}: {new.slide_id} ({change_summary})"
                )

        # Added animations
        if diff.added_animations:
            report_parts.append(f"Added {len(diff.added_animations)} animations:")
            for anim in diff.added_animations:
                text_info = f" ('{anim.element_text}')" if anim.element_text else ""
                type_info = f" [{anim.element_type}]" if anim.element_type else ""
                report_parts.append(
                    f"  - {anim.slide_id}: {anim.effect_type} on element "
                    f"{anim.element_id}{type_info}{text_info}"
                )

        # Removed animations
        if diff.removed_animations:
            report_parts.append(f"Removed {len(diff.removed_animations)} animations:")
            for anim in diff.removed_animations:
                text_info = f" ('{anim.element_text}')" if anim.element_text else ""
                type_info = f" [{anim.element_type}]" if anim.element_type else ""
                report_parts.append(
                    f"  - {anim.slide_id}: {anim.effect_type} on element "
                    f"{anim.element_id}{type_info}{text_info}"
                )

        # Modified animations
        if diff.modified_animations:
            report_parts.append(f"Modified {len(diff.modified_animations)} animations:")
            for old, new in diff.modified_animations:
                text_info = f" ('{new.element_text}')" if new.element_text else ""
                type_info = f" [{new.element_type}]" if new.element_type else ""
                report_parts.append(
                    f"  - {old.slide_id}: {old.effect_type} → "
                    f"{new.effect_type}{type_info}{text_info}"
                )

        # Added transitions
        if diff.added_transitions:
            report_parts.append(f"Added {len(diff.added_transitions)} transitions:")
            for trans in diff.added_transitions:
                report_parts.append(
                    f"  - {trans.slide_id}: {trans.transition_type} ({trans.duration}s)"
                )

        # Removed transitions
        if diff.removed_transitions:
            report_parts.append(f"Removed {len(diff.removed_transitions)} transitions:")
            for trans in diff.removed_transitions:
                report_parts.append(f"  - {trans.slide_id}: {trans.transition_type}")

        # Modified transitions
        if diff.modified_transitions:
            report_parts.append(f"Modified {len(diff.modified_transitions)} transitions:")
            for old, new in diff.modified_transitions:
                report_parts.append(
                    f"  - {old.slide_id}: {old.transition_type} → {new.transition_type}"
                )

        return (
            "\n".join(report_parts)
            if report_parts
            else "No slide, animation or transition changes detected."
        )

    def generate_detailed_diff_report(self, diff: PowerPointDiff) -> str:
        """Generate a detailed diff report with full animation information"""
        report_parts = []

        # Added slides
        if diff.added_slides:
            report_parts.append(f"=== ADDED SLIDES ({len(diff.added_slides)}) ===")
            for slide in diff.added_slides:
                report_parts.append(f"Slide {slide.slide_number}: {slide.slide_id}")
                report_parts.append(f"  Title: {slide.title or 'No title'}")
                report_parts.append(f"  Layout: {slide.layout_type or 'Unknown'}")
                report_parts.append(f"  Element Count: {slide.element_count}")
                report_parts.append(f"  Notes: {slide.notes or 'No notes'}")
                report_parts.append("")

        # Removed slides
        if diff.removed_slides:
            report_parts.append(f"=== REMOVED SLIDES ({len(diff.removed_slides)}) ===")
            for slide in diff.removed_slides:
                report_parts.append(f"Slide {slide.slide_number}: {slide.slide_id}")
                report_parts.append(f"  Title: {slide.title or 'No title'}")
                report_parts.append(f"  Layout: {slide.layout_type or 'Unknown'}")
                report_parts.append(f"  Element Count: {slide.element_count}")
                report_parts.append(f"  Notes: {slide.notes or 'No notes'}")
                report_parts.append("")

        # Modified slides
        if diff.modified_slides:
            report_parts.append(f"=== MODIFIED SLIDES ({len(diff.modified_slides)}) ===")
            for old, new in diff.modified_slides:
                report_parts.append(f"Slide {new.slide_number}: {new.slide_id}")
                report_parts.append(
                    f"  Title: {old.title or 'No title'} → {new.title or 'No title'}"
                )
                report_parts.append(
                    f"  Layout: {old.layout_type or 'Unknown'} → {new.layout_type or 'Unknown'}"
                )
                report_parts.append(f"  Element Count: {old.element_count} → {new.element_count}")
                report_parts.append(
                    f"  Notes: {old.notes or 'No notes'} → {new.notes or 'No notes'}"
                )
                report_parts.append("")

        # Added animations
        if diff.added_animations:
            report_parts.append(f"=== ADDED ANIMATIONS ({len(diff.added_animations)}) ===")
            for anim in diff.added_animations:
                report_parts.append(f"Slide: {anim.slide_id}")
                report_parts.append(f"  Element ID: {anim.element_id}")
                report_parts.append(f"  Element Type: {anim.element_type or 'Unknown'}")
                report_parts.append(f"  Element Text: {anim.element_text or 'No text'}")
                report_parts.append(f"  Effect Type: {anim.effect_type}")
                report_parts.append(f"  Trigger: {anim.trigger}")
                report_parts.append(f"  Delay: {anim.delay}s")
                report_parts.append(f"  Duration: {anim.duration}s")
                report_parts.append(f"  Order: {anim.order}")
                report_parts.append("")

        # Removed animations
        if diff.removed_animations:
            report_parts.append(f"=== REMOVED ANIMATIONS ({len(diff.removed_animations)}) ===")
            for anim in diff.removed_animations:
                report_parts.append(f"Slide: {anim.slide_id}")
                report_parts.append(f"  Element ID: {anim.element_id}")
                report_parts.append(f"  Element Type: {anim.element_type or 'Unknown'}")
                report_parts.append(f"  Element Text: {anim.element_text or 'No text'}")
                report_parts.append(f"  Effect Type: {anim.effect_type}")
                report_parts.append(f"  Trigger: {anim.trigger}")
                report_parts.append(f"  Delay: {anim.delay}s")
                report_parts.append(f"  Duration: {anim.duration}s")
                report_parts.append(f"  Order: {anim.order}")
                report_parts.append("")

        # Modified animations
        if diff.modified_animations:
            report_parts.append(f"=== MODIFIED ANIMATIONS ({len(diff.modified_animations)}) ===")
            for old, new in diff.modified_animations:
                report_parts.append(f"Slide: {old.slide_id}")
                report_parts.append(f"  Element ID: {old.element_id}")
                report_parts.append(f"  Element Type: {new.element_type or 'Unknown'}")
                report_parts.append(f"  Element Text: {new.element_text or 'No text'}")
                report_parts.append(f"  Effect Type: {old.effect_type} → {new.effect_type}")
                report_parts.append(f"  Trigger: {old.trigger} → {new.trigger}")
                report_parts.append(f"  Delay: {old.delay}s → {new.delay}s")
                report_parts.append(f"  Duration: {old.duration}s → {new.duration}s")
                report_parts.append("")

        # Transitions
        if diff.added_transitions or diff.removed_transitions or diff.modified_transitions:
            report_parts.append("=== TRANSITIONS ===")

            if diff.added_transitions:
                report_parts.append(f"Added {len(diff.added_transitions)} transitions:")
                for trans in diff.added_transitions:
                    report_parts.append(
                        f"  - {trans.slide_id}: {trans.transition_type} ({trans.duration}s)"
                    )

            if diff.removed_transitions:
                report_parts.append(f"Removed {len(diff.removed_transitions)} transitions:")
                for trans in diff.removed_transitions:
                    report_parts.append(f"  - {trans.slide_id}: {trans.transition_type}")

            if diff.modified_transitions:
                report_parts.append(f"Modified {len(diff.modified_transitions)} transitions:")
                for old, new in diff.modified_transitions:
                    report_parts.append(
                        f"  - {old.slide_id}: {old.transition_type} → {new.transition_type}"
                    )

        return (
            "\n".join(report_parts)
            if report_parts
            else "No slide, animation or transition changes detected."
        )


# Example usage
def main():
    evaluator = PowerPointDiffEvaluator()

    # Compare two PowerPoint files
    diff = evaluator.compare_files(
        "/Users/apurvag/Downloads/Presentation1.pptx",
        "/Users/apurvag/Downloads/Presentation1-modified.pptx",
    )

    # Generate concise report
    print("=== CONCISE DIFF REPORT ===")
    concise_report = evaluator.generate_concise_diff_report(diff)
    print(concise_report)

    # Generate detailed report
    print("\n=== DETAILED DIFF REPORT ===")
    detailed_report = evaluator.generate_detailed_diff_report(diff)
    print(detailed_report)

    # Get structured data for LLM processing
    diff_data = diff.to_dict()
    print("\n=== STRUCTURED DIFF DATA ===")
    print(json.dumps(diff_data, indent=2))

    # Summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total slides added: {len(diff.added_slides)}")
    print(f"Total slides removed: {len(diff.removed_slides)}")
    print(f"Total slides modified: {len(diff.modified_slides)}")
    print(f"Total animations added: {len(diff.added_animations)}")
    print(f"Total animations removed: {len(diff.removed_animations)}")
    print(f"Total animations modified: {len(diff.modified_animations)}")
    print(f"Total transitions added: {len(diff.added_transitions)}")
    print(f"Total transitions removed: {len(diff.removed_transitions)}")
    print(f"Total transitions modified: {len(diff.modified_transitions)}")

    # Slide breakdown by layout type
    if diff.added_slides:
        print("\n=== ADDED SLIDES BY LAYOUT TYPE ===")
        layout_counts = {}
        for slide in diff.added_slides:
            layout_type = slide.layout_type or "Unknown"
            layout_counts[layout_type] = layout_counts.get(layout_type, 0) + 1

        for layout_type, count in sorted(layout_counts.items()):
            print(f"  {layout_type}: {count}")

    # Animation breakdown by element type
    if diff.added_animations:
        print("\n=== ADDED ANIMATIONS BY ELEMENT TYPE ===")
        type_counts = {}
        for anim in diff.added_animations:
            elem_type = anim.element_type or "unknown"
            type_counts[elem_type] = type_counts.get(elem_type, 0) + 1

        for elem_type, count in sorted(type_counts.items()):
            print(f"  {elem_type}: {count}")

    # Animation breakdown by effect type
    if diff.added_animations:
        print("\n=== ADDED ANIMATIONS BY EFFECT TYPE ===")
        effect_counts = {}
        for anim in diff.added_animations:
            effect_counts[anim.effect_type] = effect_counts.get(anim.effect_type, 0) + 1

        for effect_type, count in sorted(effect_counts.items()):
            print(f"  {effect_type}: {count}")


if __name__ == "__main__":
    main()
