#!/usr/bin/env python3
"""Locked biological acquisition profiles for the v4 trajectory pipeline.

The internal channel names ``green``, ``red`` and ``purple`` are retained only
because the Fiji/MATLAB pipeline uses those filenames.  Biological identity,
raw-channel index, fluorophore and the anchor are owned by the selected profile.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import tifffile


@dataclass(frozen=True)
class ChannelSpec:
    corrected_channel: str
    prefix: str
    raw_index: int
    marker: str
    marker_slug: str
    site_id: str
    genomic_locus: str
    fluorophore: str
    display_color: str


@dataclass(frozen=True)
class ExperimentProfile:
    name: str
    description: str
    allowed_channel_counts: tuple[int, ...]
    raw_channel_name_tokens: tuple[str, ...]
    filename_required_tokens: tuple[str, ...]
    filename_any_tokens: tuple[str, ...]
    nucleus_raw_index: int | None
    alignment_channel: str
    alignment_raw_index: int
    anchor_channel: str
    channels: tuple[ChannelSpec, ...]
    scientific_scope: str

    def channel(self, corrected_channel: str) -> ChannelSpec:
        for spec in self.channels:
            if spec.corrected_channel == corrected_channel:
                return spec
        raise KeyError(f"Profile {self.name!r} has no channel {corrected_channel!r}")

    def channel_from_prefix(self, prefix: str) -> ChannelSpec:
        for spec in self.channels:
            if spec.prefix == prefix:
                return spec
        raise KeyError(f"Profile {self.name!r} has no prefix {prefix!r}")

    @property
    def anchor(self) -> ChannelSpec:
        return self.channel(self.anchor_channel)

    def to_manifest(self) -> dict:
        return asdict(self)

    def validate_crop(self, path: Path) -> dict:
        path = path.resolve()
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            axes = series.axes
            shape = tuple(int(value) for value in series.shape)
        if axes not in ("TCYX", "TZCYX"):
            raise ValueError(f"axes={axes}, expected TCYX or TZCYX")
        channel_count = shape[axes.index("C")]
        if channel_count not in self.allowed_channel_counts:
            expected = "/".join(str(value) for value in self.allowed_channel_counts)
            raise ValueError(
                f"profile={self.name} requires C={expected}, found C={channel_count} "
                f"in axes={axes}, shape={shape}"
            )

        sidecar_path = path.with_name(path.stem + "_metadata.json")
        if not sidecar_path.is_file():
            raise ValueError(
                f"profile={self.name} requires the Step-3 acquisition sidecar: "
                f"{sidecar_path.name}"
            )
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(f"invalid acquisition sidecar {sidecar_path}: {exc}") from exc
        sidecar_count = int(sidecar.get("crop_shape", {}).get("C", -1))
        if sidecar_count != channel_count:
            raise ValueError(
                f"profile={self.name} sidecar/TIFF channel mismatch: "
                f"sidecar C={sidecar_count}, TIFF C={channel_count}"
            )
        channels = sidecar.get("channels")
        if not isinstance(channels, list) or len(channels) != channel_count:
            raise ValueError(
                f"profile={self.name} expected {channel_count} sidecar channel rows, "
                f"found {0 if not isinstance(channels, list) else len(channels)}"
            )
        observed_names = []
        for raw_index, expected_token in enumerate(self.raw_channel_name_tokens):
            row = channels[raw_index]
            if int(row.get("index", -1)) != raw_index:
                raise ValueError(
                    f"profile={self.name} sidecar channel index mismatch at C{raw_index}: {row}"
                )
            observed_name = str(row.get("name", ""))
            observed_names.append(observed_name)
            if expected_token.casefold() not in observed_name.casefold():
                raise ValueError(
                    f"profile={self.name} raw C{raw_index} requires channel-name token "
                    f"{expected_token!r}, found {observed_name!r}"
                )

        evidence = " ".join(
            str(sidecar.get(key, "")) for key in ("source_nd2", "stem")
        )
        normalized_name = f"{path.name} {evidence}".casefold()
        missing = [
            token for token in self.filename_required_tokens
            if token.casefold() not in normalized_name
        ]
        if missing:
            raise ValueError(
                f"profile={self.name} filename evidence failed; missing token(s) "
                f"{missing} from {path.name!r}"
            )
        if self.filename_any_tokens and not any(
            token.casefold() in normalized_name for token in self.filename_any_tokens
        ):
            raise ValueError(
                f"profile={self.name} requires at least one filename token from "
                f"{self.filename_any_tokens}, found {path.name!r}"
            )
        return {
            "path": str(path),
            "axes": axes,
            "shape": shape,
            "channel_count": channel_count,
            "sidecar": str(sidecar_path.resolve()),
            "raw_channel_names": observed_names,
            "profile": self.name,
            "anchor_channel": self.anchor_channel,
            "anchor_raw_index": self.anchor.raw_index,
            "anchor_marker": self.anchor.marker,
        }


PROFILES = {
    "chr3_sites_2_3_4": ExperimentProfile(
        name="chr3_sites_2_3_4",
        description=(
            "Oligo-LiveFISH Chr3 sites 2/3/4: Site 2=A488, Site 3=A565, "
            "Site 4=A647, plus Hoechst nucleus"
        ),
        allowed_channel_counts=(4,),
        raw_channel_name_tokens=("405", "640", "488", "561"),
        filename_required_tokens=("chr3", "195M", "195.7M", "198M", "488", "565", "647"),
        filename_any_tokens=(),
        nucleus_raw_index=0,
        alignment_channel="nucleus",
        alignment_raw_index=0,
        anchor_channel="green",
        channels=(
            ChannelSpec(
                corrected_channel="green",
                prefix="G",
                raw_index=2,
                marker="Chr3 Site 2 (195 Mb, A488)",
                marker_slug="chr3_site2_195m_a488",
                site_id="site2",
                genomic_locus="chr3:195M",
                fluorophore="A488",
                display_color="#00A65A",
            ),
            ChannelSpec(
                corrected_channel="red",
                prefix="R",
                raw_index=1,
                marker="Chr3 Site 4 (198 Mb, A647)",
                marker_slug="chr3_site4_198m_a647",
                site_id="site4",
                genomic_locus="chr3:198M",
                fluorophore="A647",
                display_color="#D62728",
            ),
            ChannelSpec(
                corrected_channel="purple",
                prefix="P",
                raw_index=3,
                marker="Chr3 Site 3 (195.7 Mb, A565)",
                marker_slug="chr3_site3_195p7m_a565",
                site_id="site3",
                genomic_locus="chr3:195.7M",
                fluorophore="A565",
                display_color="#F4B400",
            ),
        ),
        scientific_scope=(
            "Figure-6-style three-locus Chr3 analysis and manuscript/ML comparison. "
            "The 488-nm Site 2 channel is the locked reference anchor."
        ),
    ),
    "dsb_53bp1_site1_site2": ExperimentProfile(
        name="dsb_53bp1_site1_site2",
        description="Three-channel DSB acquisition: 53BP1, Site 1 and Site 2",
        allowed_channel_counts=(3,),
        raw_channel_name_tokens=("GFP", "RFP", "Cy5"),
        filename_required_tokens=(),
        filename_any_tokens=("DSB", "53BP1"),
        nucleus_raw_index=None,
        alignment_channel="green",
        alignment_raw_index=0,
        anchor_channel="purple",
        channels=(
            ChannelSpec(
                corrected_channel="green",
                prefix="G",
                raw_index=0,
                marker="53BP1 (Green)",
                marker_slug="53bp1",
                site_id="53bp1",
                genomic_locus="",
                fluorophore="GFP/green acquisition",
                display_color="#00A65A",
            ),
            ChannelSpec(
                corrected_channel="red",
                prefix="R",
                raw_index=1,
                marker="Site 1 (Yellow)",
                marker_slug="site1",
                site_id="site1",
                genomic_locus="",
                fluorophore="yellow acquisition",
                display_color="#F4B400",
            ),
            ChannelSpec(
                corrected_channel="purple",
                prefix="P",
                raw_index=2,
                marker="Site 2 (Purple)",
                marker_slug="site2",
                site_id="site2",
                genomic_locus="",
                fluorophore="purple acquisition",
                display_color="#7A3DB8",
            ),
        ),
        scientific_scope=(
            "DSB/53BP1 analysis only. Site 2 is the locked reference anchor; "
            "the synthetic nucleus scaffold is not valid for morphology features."
        ),
    ),
}


def profile_choices() -> tuple[str, ...]:
    return tuple(PROFILES)


def get_profile(name: str) -> ExperimentProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown experiment profile {name!r}; choose one of {profile_choices()}"
        ) from exc
