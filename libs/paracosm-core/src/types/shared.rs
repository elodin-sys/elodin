use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
};

#[derive(Clone, Debug)]
pub struct SharedNum<T> {
    storage: Arc<AtomicU64>,
    _phantom: PhantomData<T>,
}

impl<T: ToU64Storage + Default> Default for SharedNum<T> {
    fn default() -> Self {
        Self {
            storage: Arc::new(AtomicU64::new(T::default().to_bits())),
            _phantom: Default::default(),
        }
    }
}

impl<T: ToU64Storage> SharedNum<T> {
    pub fn load(&self) -> SharedNumRef<'_, T> {
        let storage = self.storage.load(Ordering::SeqCst);
        SharedNumRef {
            storage: self.storage.as_ref(),
            num: T::from_bits(storage),
        }
    }
}

pub struct SharedNumRef<'a, T: ToU64Storage> {
    storage: &'a AtomicU64,
    num: T,
}

impl<'a, T: ToU64Storage> Deref for SharedNumRef<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.num
    }
}

impl<'a, T: ToU64Storage> DerefMut for SharedNumRef<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.num
    }
}

impl<'a, T: ToU64Storage> Drop for SharedNumRef<'a, T> {
    fn drop(&mut self) {
        self.storage.store(self.num.to_bits(), Ordering::SeqCst);
    }
}

pub trait ToU64Storage {
    fn from_bits(num: u64) -> Self;
    fn to_bits(&self) -> u64;
}

impl ToU64Storage for f64 {
    fn from_bits(num: u64) -> Self {
        f64::from_bits(num)
    }

    fn to_bits(&self) -> u64 {
        f64::to_bits(*self)
    }
}

#[derive(Clone, Debug)]
pub struct ObservableNum<T> {
    storage: Arc<ObservableNumInner>,
    _phantom: PhantomData<T>,
}
#[derive(Debug)]
struct ObservableNumInner {
    num: AtomicU64,
    changed: AtomicBool,
}

impl<T: ToU64Storage + Default> Default for ObservableNum<T> {
    fn default() -> Self {
        Self {
            storage: Arc::new(ObservableNumInner {
                num: AtomicU64::new(T::default().to_bits()),
                changed: AtomicBool::new(false),
            }),
            _phantom: Default::default(),
        }
    }
}

impl<T: ToU64Storage> ObservableNum<T> {
    pub fn load(&self) -> ObservableNumRef<'_, T> {
        let storage = self.storage.num.load(Ordering::SeqCst);
        ObservableNumRef {
            storage: self.storage.as_ref(),
            num: T::from_bits(storage),
        }
    }

    pub fn has_changed(&self) -> bool {
        self.storage.changed.swap(false, Ordering::SeqCst)
    }
}

pub struct ObservableNumRef<'a, T: ToU64Storage> {
    storage: &'a ObservableNumInner,
    num: T,
}

impl<'a, T: ToU64Storage> Deref for ObservableNumRef<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.num
    }
}

impl<'a, T: ToU64Storage> DerefMut for ObservableNumRef<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.num
    }
}

impl<'a, T: ToU64Storage> Drop for ObservableNumRef<'a, T> {
    fn drop(&mut self) {
        let bits = self.num.to_bits();
        let old = self.storage.num.swap(bits, Ordering::SeqCst);
        self.storage.changed.store(old != bits, Ordering::SeqCst);
    }
}
