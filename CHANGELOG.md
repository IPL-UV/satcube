# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Enhanced cloud masking functionality for Sentinel-2 images.
- Support for gap-filling and super-resolution.
- Added methods for generating monthly composites and smoothing reflectance values.
- Integration with Google Earth Engine for image retrieval and cloud removal.

### Changed
- Optimized data handling for large image collections.
- Updated README with new examples and documentation on image interpolation.

### Fixed
- Fixed minor issues with cloud cover threshold handling.
- Resolved bug with image display function.

## [0.1.0] - 2024-07-07

### Added
- Initial release of **satcube** with core functionalities for downloading, cloud masking, and processing Sentinel-2 satellite image cubes.
- Basic functionality for querying and filtering Sentinel-2 metadata.
- Methods for creating monthly composites from image collections.
- Added support for interpolation of missing data and reflectance value smoothing.