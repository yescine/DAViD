"""Helper script to download the SynthHuman dataset.

This python file is licensed under the MIT license (see below).
The SynthHuman dataset is licensed under the CDLA-2.0  license (https://cdla.dev/permissive-2-0/).

Copyright (c) Microsoft Corporation.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional
from zipfile import ZipFile

N_PARTS = 60


def extract(data_path: Path, out_path: Optional[Path] = None) -> None:
    """Extract the data from the given path."""
    print(f"Extracting {data_path.name}...")
    if data_path.suffix == ".zip":
        out_path = out_path or data_path.parent / data_path.stem
        with ZipFile(data_path) as f:
            f.extractall(out_path)
    else:
        raise ValueError(f"Unknown file type {data_path.suffix}")


def download_synthhuman_data(data_dir: Path, single_sample: bool, single_chunck: bool) -> None:
    """Download the SynthHuman dataset."""
    data_dir.mkdir(exist_ok=True, parents=True)
    zip_dir = data_dir / "SynthHuman_zip"
    zip_dir.mkdir(exist_ok=True, parents=True)
    parts = (
        ["SynthHuman_sample.zip"]
        if single_sample
        else [f"SynthHuman_{i:04d}.zip" for i in range(0, 1 if single_chunck else N_PARTS)]
    )
    for part in parts:
        out_path = zip_dir / part
        print(f"Downloading {part}...")
        url = f"https://facesyntheticspubwedata.z6.web.core.windows.net/iccv-2025/data/{part}"
        try:
            subprocess.check_call(
                [
                    "wget",
                    url,
                    "-O",
                    str(out_path),
                    "--no-check-certificate",
                    "--continue",
                    "--secure-protocol=TLSv1_2",
                ]
            )
        except FileNotFoundError as exc:
            raise RuntimeError("wget not found, please install it") from exc
        except subprocess.CalledProcessError:
            print("Download failed")
            if out_path.exists():
                out_path.unlink()
            sys.exit(1)
        extract(out_path, data_dir / "SynthHuman")
        out_path.unlink()
    zip_dir.rmdir()


def main() -> None:
    """Download and unpack the dataset."""
    parser = argparse.ArgumentParser(description="Download SynthHuman dataset")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument(
        "--single-sample",
        action="store_true",
        help="Only download one subject from the dataset",
    )
    parser.add_argument(
        "--single-chunk",
        action="store_true",
        help="Only download one chunk from the dataset",
    )
    args = parser.parse_args()
    assert not (args.single_sample and args.single_chunk), "Cannot specify both single-sample and single-chunk"
    data_dir = Path(args.output_dir)
    download_synthhuman_data(data_dir, args.single_sample, args.single_chunk)


if __name__ == "__main__":
    main()
