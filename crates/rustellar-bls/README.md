# rustellar-bls

Box least squares transit search (Kovacs, Zucker & Mazeh 2002) in pure Rust,
part of [rustellar](../../README.md).

**Status:** in development. So far: the checked search inputs (`Durations`,
`Periods`), astropy's automatic period grid (`AutoPeriod`, the same periods to
the last bit), Ofir's (2014) optimal period grid (`OfirPeriod`), and one box
fitted exactly (`TransitBox`, `BoxFit`), the exact search at one period and
duration over every mid-transit time (`TransitBox::best`), and the search
over many periods (`Bls`, `SearchOptions`, `BlsResult`): binned at every
period on all cores, with the highest peaks refined exactly. Vetting
statistics are not written yet.

## License

MIT or Apache-2.0, at your option.
