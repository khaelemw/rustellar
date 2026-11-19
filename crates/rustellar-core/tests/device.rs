// Where a search runs: a GPU where one can be used by default.

use rustellar_core::Device;

#[test]
fn the_default_is_auto() {
    assert_eq!(Device::default(), Device::Auto);
    assert_ne!(Device::Cpu, Device::Gpu(0));
    assert_eq!(format!("{:?}", Device::Gpu(1)), "Gpu(1)");
}
