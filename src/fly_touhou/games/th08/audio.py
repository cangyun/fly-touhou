"""Set original TH08's audio options in a private game's configuration."""

import struct


CONFIG_SIZE = 0x3C
CONFIG_VERSION = 0x00080001
MUSIC_MODE = 0x1F
PLAY_SOUNDS = 0x20
MUSIC_VOLUME = 0x27
SOUND_VOLUME = 0x28


def _native_default():
    data = bytearray(CONFIG_SIZE)
    struct.pack_into("<10hIhh", data, 0, 0, 1, 2, 4, -1, -1, -1, -1, 3, 0,
                     CONFIG_VERSION, 600, 600)
    data[0x1C:0x29] = bytes((2, 3, 0, 1, 1, 1, 0, 0, 2, 0, 0, 100, 80))
    data[0x38] = 1
    return data


def _read_config(path):
    if not path.is_file():
        return None
    data = path.read_bytes()
    if len(data) != CONFIG_SIZE or struct.unpack_from("<I", data, 0x14)[0] != CONFIG_VERSION:
        return None
    # Invalid options cause the original to regenerate an audible default.
    choice_limits = (7, 4, 2, 3, 2, 6, 2, 3, 3, 2, 2)
    if any(value >= limit for value, limit in zip(data[0x1C:0x27], choice_limits)):
        return None
    return data


def configure_private_audio(source, local, *, recording=False):
    """Copy native options on first use, then change only BGM and SFX settings."""
    if source.resolve() == local.resolve():
        raise ValueError("TH08 audio options require a private game copy")
    original = _read_config(source) or _native_default()
    current = _read_config(local)
    config = bytearray(current or original)
    if recording:
        config[MUSIC_MODE] = original[MUSIC_MODE] or 1
        config[PLAY_SOUNDS] = 1
        config[MUSIC_VOLUME] = original[MUSIC_VOLUME] if 0 < original[MUSIC_VOLUME] <= 100 else 100
        config[SOUND_VOLUME] = original[SOUND_VOLUME] if 0 < original[SOUND_VOLUME] <= 100 else 80
    else:
        config[MUSIC_MODE] = 0
        config[PLAY_SOUNDS] = 0
    if config != current:
        temporary = local.with_suffix(".cfg.copying")
        temporary.write_bytes(config)
        temporary.replace(local)
