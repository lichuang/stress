#![feature(associated_type_bounds)]

use std::{
    fmt,
    path::PathBuf,
    process,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    thread,
};

#[cfg(feature = "dh")]
use dhat::{Dhat, DhatAlloc};

use num_format::{Locale, ToFormattedString};
use rand::{thread_rng, Rng};

#[cfg(feature = "jemalloc")]
mod alloc {
    use jemallocator::Jemalloc;
    use std::alloc::Layout;

    #[global_allocator]
    static ALLOCATOR: Jemalloc = Jemalloc;
}

#[cfg(feature = "memshred")]
mod alloc {
    use std::alloc::{Layout, System};

    #[global_allocator]
    static ALLOCATOR: Alloc = Alloc;

    #[derive(Default, Debug, Clone, Copy)]
    struct Alloc;

    unsafe impl std::alloc::GlobalAlloc for Alloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let ret = System.alloc(layout);
            assert_ne!(ret, std::ptr::null_mut());
            std::ptr::write_bytes(ret, 0xa1, layout.size());
            ret
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            std::ptr::write_bytes(ptr, 0xde, layout.size());
            System.dealloc(ptr, layout)
        }
    }
}

#[cfg(feature = "measure_allocs")]
mod alloc {
    use std::alloc::{Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering::Release};

    pub static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
    pub static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);

    #[global_allocator]
    static ALLOCATOR: Alloc = Alloc;

    #[derive(Default, Debug, Clone, Copy)]
    struct Alloc;

    unsafe impl std::alloc::GlobalAlloc for Alloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            ALLOCATIONS.fetch_add(1, Release);
            ALLOCATED_BYTES.fetch_add(layout.size(), Release);
            System.alloc(layout)
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            System.dealloc(ptr, layout)
        }
    }
}

#[global_allocator]
#[cfg(feature = "dh")]
static ALLOCATOR: DhatAlloc = DhatAlloc;

static TOTAL: AtomicUsize = AtomicUsize::new(0);
static GET_TOTAL: AtomicUsize = AtomicUsize::new(0);
static SET_TOTAL: AtomicUsize = AtomicUsize::new(0);
static DEL_TOTAL: AtomicUsize = AtomicUsize::new(0);
static CAS_TOTAL: AtomicUsize = AtomicUsize::new(0);
static MERGE_TOTAL: AtomicUsize = AtomicUsize::new(0);

const USAGE: &str = "
Usage: stress [--threads=<#>] [--burn-in] [--duration=<s>] \
    [--key-len=<l>] [--val-len=<l>] \
    [--get-prop=<p>] \
    [--set-prop=<p>] \
    [--del-prop=<p>] \
    [--cas-prop=<p>] \
    [--scan-prop=<p>] \
    [--merge-prop=<p>] \
    [--entries=<n>] \
    [--sequential] \
    [--total-ops=<n>] \
    [--flush-every=<ms>]

Options:
    --kind=<#>         Kind of db, sled:0, rocksdb:1 [default:0].
    --threads=<#>      Number of threads [default: 4].
    --burn-in          Don't halt until we receive a signal.
    --duration=<s>     Seconds to run for [default: 10].
    --key-len=<l>      The length of keys [default: 10].
    --val-len=<l>      The length of values [default: 100].
    --get-prop=<p>     The relative proportion of get requests [default: 94].
    --set-prop=<p>     The relative proportion of set requests [default: 2].
    --del-prop=<p>     The relative proportion of del requests [default: 1].
    --cas-prop=<p>     The relative proportion of cas requests [default: 1].
    --scan-prop=<p>    The relative proportion of scan requests [default: 1].
    --merge-prop=<p>   The relative proportion of merge requests [default: 1].
    --entries=<n>      The total keyspace [default: 100000].
    --sequential       Run the test in sequential mode instead of random.
    --total-ops=<n>    Stop test after executing a total number of operations.
    --flush-every=<m>  Flush and sync the database every ms [default: 200].
    --cache-mb=<mb>    Size of the page cache in megabytes [default: 1024].
";

#[derive(Debug, Clone, Copy)]
pub struct Args {
    threads: usize,
    burn_in: bool,
    duration: u64,
    key_len: usize,
    val_len: usize,
    get_prop: usize,
    set_prop: usize,
    del_prop: usize,
    cas_prop: usize,
    scan_prop: usize,
    merge_prop: usize,
    entries: usize,
    sequential: bool,
    total_ops: Option<usize>,
    flush_every: u64,
    cache_mb: usize,
    kind: usize,
}

impl Default for Args {
    fn default() -> Args {
        Args {
            threads: 4,
            burn_in: false,
            duration: 10,
            key_len: 10,
            val_len: 100,
            get_prop: 94,
            set_prop: 2,
            del_prop: 1,
            cas_prop: 1,
            scan_prop: 1,
            merge_prop: 1,
            entries: 100000,
            sequential: false,
            total_ops: None,
            flush_every: 200,
            cache_mb: 1024,
            kind: 0,
        }
    }
}

fn parse<'a, I, T>(mut iter: I) -> T
where
    I: Iterator<Item = &'a str>,
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    iter.next().expect(USAGE).parse().expect(USAGE)
}

impl Args {
    fn parse() -> Args {
        let mut args = Args::default();
        for raw_arg in std::env::args().skip(1) {
            let mut splits = raw_arg[2..].split('=');
            match splits.next().unwrap() {
                "threads" => args.threads = parse(&mut splits),
                "burn-in" => args.burn_in = true,
                "duration" => args.duration = parse(&mut splits),
                "key-len" => args.key_len = parse(&mut splits),
                "val-len" => args.val_len = parse(&mut splits),
                "get-prop" => args.get_prop = parse(&mut splits),
                "set-prop" => args.set_prop = parse(&mut splits),
                "del-prop" => args.del_prop = parse(&mut splits),
                "cas-prop" => args.cas_prop = parse(&mut splits),
                "scan-prop" => args.scan_prop = parse(&mut splits),
                "merge-prop" => args.merge_prop = parse(&mut splits),
                "entries" => args.entries = parse(&mut splits),
                "sequential" => args.sequential = true,
                "total-ops" => args.total_ops = Some(parse(&mut splits)),
                "flush-every" => args.flush_every = parse(&mut splits),
                "cache-mb" => args.cache_mb = parse(&mut splits),
                "kind" => args.kind = parse(&mut splits),
                "help" => {
                    println!("USAGE: {}", USAGE);
                    process::exit(0);
                }
                other => panic!("unknown option: {}, {}", other, USAGE),
            }
        }
        args
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    message: String,
}

impl std::error::Error for Error {
    fn description(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        self.message.fmt(formatter)
    }
}

impl From<sled::Error> for Error {
    fn from(e: sled::Error) -> Error {
        Self {
            message: format!("sled error: {}", e),
        }
    }
}

impl From<sled::CompareAndSwapError> for Error {
    fn from(e: sled::CompareAndSwapError) -> Error {
        Self {
            message: format!("sled error: {}", e),
        }
    }
}

impl From<rocksdb::Error> for Error {
    fn from(e: rocksdb::Error) -> Error {
        Self {
            message: format!("rocksdb error: {}", e),
        }
    }
}

enum Db {
    SledDb(sled::Db),
    RocksDb(rocksdb::DB),
}

fn new_sled_db(_path: String, args: &Args) -> sled::Db {
    let config = sled::Config::new()
        .cache_capacity(args.cache_mb * 1024 * 1024)
        .flush_every_ms(if args.flush_every == 0 {
            None
        } else {
            Some(args.flush_every)
        });

    let tree = config.open().unwrap();
    tree.set_merge_operator(concatenate_merge);
    tree
}

fn new_rocks_db(path: String, _args: &Args) -> rocksdb::DB {
    let path = PathBuf::from(path);
    let db = rocksdb::DB::open_default(path.to_str().unwrap()).unwrap();
    db
}

impl Db {
    fn get<K: AsRef<[u8]>>(&self, key: K) -> Result<Option<Vec<u8>>, Error> {
        match self {
            Db::SledDb(tree) => {
                let ret = tree.get(key)?;
                Ok(ret.map(|vec| vec.to_vec()))
            }
            Db::RocksDb(db) => {
                let ret = db.get(key)?;
                Ok(ret)
            }
        }
    }

    fn insert<K: AsRef<[u8]>>(&self, key: K, value: Vec<u8>) -> Result<(), Error> {
        match self {
            Db::SledDb(tree) => {
                tree.insert(&key, value)?;
                Ok(())
            }
            Db::RocksDb(db) => {
                db.put(key, value)?;
                Ok(())
            }
        }
    }

    fn remove<K: AsRef<[u8]>>(&self, key: K) -> Result<(), Error> {
        match self {
            Db::SledDb(tree) => {
                tree.remove(key)?;
                Ok(())
            }
            Db::RocksDb(db) => {
                db.delete(key)?;
                Ok(())
            }
        }
    }

    fn compare_and_swap<K: AsRef<[u8]>>(
        &self,
        key: K,
        old: Option<Vec<u8>>,
        new: Option<Vec<u8>>,
    ) -> Result<(), Error> {
        match self {
            Db::SledDb(tree) => {
                let _ = tree.compare_and_swap(&key, old, new)?;
                Ok(())
            }
            Db::RocksDb(_db) => {
                unimplemented!()
            }
        }
    }

    fn merge<K: AsRef<[u8]>>(&self, key: K, value: Vec<u8>) -> Result<Option<Vec<u8>>, Error> {
        match self {
            Db::SledDb(tree) => {
                let ret = tree.merge(key, value)?;
                Ok(ret.map(|vec| vec.to_vec()))
            }
            Db::RocksDb(_db) => {
                unimplemented!()
            }
        }
    }
}

/*
#[derive(Clone)]
struct RocksDb {
    db: Arc<rocksdb::DB>,
}

impl RocksDb {
    pub fn new(path: String, _args: &Args) -> Self {
        let path = PathBuf::from(path);
        let db = Arc::new(rocksdb::DB::open_default(path.to_str().unwrap()).unwrap());

        Self { db }
    }
}

impl<K> Db<K> for RocksDb {
    fn get(&self, key: K) -> Result<Option<Vec<u8>>, Error> {
        let ret = self.db.get(key)?;
        Ok(ret)
    }

    fn insert(&self, key: K, value: Vec<u8>) -> Result<(), Error> {
        self.db.put(key, value)?;
        Ok(())
    }

    fn remove(&self, key: K) -> Result<(), Error> {
        self.db.delete(key)?;
        Ok(())
    }

    fn compare_and_swap(
        &self,
        _key: K,
        _old: Option<Vec<u8>>,
        _new: Option<Vec<u8>>,
    ) -> Result<(), Error> {
        unimplemented!()
    }

    fn merge(&self, _key: K, _value: Vec<u8>) -> Result<Option<Vec<u8>>, Error> {
        unimplemented!()
    }
}
*/

fn report(shutdown: Arc<AtomicBool>) {
    let mut set_last = 0;
    let mut get_last = 0;
    let mut del_last = 0;
    let mut cas_last = 0;
    let mut merge_last = 0;
    while !shutdown.load(Ordering::Relaxed) {
        thread::sleep(std::time::Duration::from_secs(1));
        let set_total = SET_TOTAL.load(Ordering::Acquire);
        let get_total = GET_TOTAL.load(Ordering::Acquire);
        let del_total = DEL_TOTAL.load(Ordering::Acquire);
        let cas_total = CAS_TOTAL.load(Ordering::Acquire);
        let merge_total = MERGE_TOTAL.load(Ordering::Acquire);

        println!(
            "did {} set/{} get/{} del/{} cas/{} merge ops, {}mb RSS",
            (set_total - set_last).to_formatted_string(&Locale::en),
            (get_total - get_last).to_formatted_string(&Locale::en),
            (del_total - del_last).to_formatted_string(&Locale::en),
            (cas_total - cas_last).to_formatted_string(&Locale::en),
            (merge_total - merge_last).to_formatted_string(&Locale::en),
            rss() / (1024 * 1024)
        );

        set_last = set_total;
        get_last = get_total;
        del_last = del_total;
        cas_last = cas_total;
        merge_last = merge_total;
    }
}

fn concatenate_merge(
    _key: &[u8],              // the key being merged
    old_value: Option<&[u8]>, // the previous value, if one existed
    merged_bytes: &[u8],      // the new bytes being merged in
) -> Option<Vec<u8>> {
    // set the new value, return None to delete
    let mut ret = old_value.map(|ov| ov.to_vec()).unwrap_or_else(Vec::new);

    ret.extend_from_slice(merged_bytes);

    Some(ret)
}

fn run(args: Args, db: Arc<Db>, shutdown: Arc<AtomicBool>) {
    let get_max = args.get_prop;
    let set_max = get_max + args.set_prop;
    let del_max = set_max + args.del_prop;
    let cas_max = del_max + args.cas_prop;
    let merge_max = cas_max + args.merge_prop;
    let scan_max = merge_max + args.scan_prop;

    let keygen = |len| -> sled::IVec {
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let i = if args.sequential {
            SEQ.fetch_add(1, Ordering::Relaxed)
        } else {
            thread_rng().gen::<usize>()
        } % args.entries;

        let start = if len < 8 { 8 - len } else { 0 };

        let i_keygen = &i.to_be_bytes()[start..];

        i_keygen.iter().cycle().take(len).copied().collect()
    };

    let valgen = |len| -> sled::IVec {
        if len == 0 {
            return vec![].into();
        }

        let i: usize = thread_rng().gen::<usize>() % (len * 8);

        let i_keygen = i.to_be_bytes();

        i_keygen
            .iter()
            .skip_while(|v| **v == 0)
            .cycle()
            .take(len)
            .copied()
            .collect()
    };

    let mut rng = thread_rng();

    while !shutdown.load(Ordering::Relaxed) {
        let _op = TOTAL.fetch_add(1, Ordering::Release);
        let key = keygen(args.key_len);
        let choice = rng.gen_range(0, scan_max + 1);

        match choice {
            v if v <= get_max => {
                db.get(&key).unwrap();
                GET_TOTAL.fetch_add(1, Ordering::Release);
            }
            v if v > get_max && v <= set_max => {
                let value = valgen(args.val_len);
                db.insert(&key, value.to_vec()).unwrap();
                SET_TOTAL.fetch_add(1, Ordering::Release);
            }
            v if v > set_max && v <= del_max => {
                db.remove(&key).unwrap();
                DEL_TOTAL.fetch_add(1, Ordering::Release);
            }
            v if v > del_max && v <= cas_max => {
                let old = if rng.gen::<bool>() {
                    let value = valgen(args.val_len);
                    Some(value.to_vec())
                } else {
                    None
                };

                let new = if rng.gen::<bool>() {
                    let value = valgen(args.val_len);
                    Some(value.to_vec())
                } else {
                    None
                };

                if let Err(e) = db.compare_and_swap(&key, old, new) {
                    panic!("operational error: {:?}", e);
                }
                CAS_TOTAL.fetch_add(1, Ordering::Release);
            }
            v if v > cas_max && v <= merge_max => {
                let value = valgen(args.val_len);
                db.merge(&key, value.to_vec()).unwrap();
                MERGE_TOTAL.fetch_add(1, Ordering::Release);
            }
            _ => {
                /*
                let iter = tree.range(key..).map(|res| res.unwrap());

                if op % 2 == 0 {
                    let _ = iter.take(rng.gen_range(0, 15)).collect::<Vec<_>>();
                } else {
                    let _ = iter.rev().take(rng.gen_range(0, 15)).collect::<Vec<_>>();
                }
                */
            }
        }
    }
}

fn rss() -> usize {
    #[cfg(target_os = "linux")]
    {
        use std::io::prelude::*;
        use std::io::BufReader;

        let mut buf = String::new();
        let mut f = BufReader::new(std::fs::File::open("/proc/self/statm").unwrap());
        f.read_line(&mut buf).unwrap();
        let mut parts = buf.split_whitespace();
        let rss_pages = parts.nth(1).unwrap().parse::<usize>().unwrap();
        rss_pages * 4096
    }
    #[cfg(not(target_os = "linux"))]
    {
        0
    }
}

fn main() {
    #[cfg(feature = "logging")]
    setup_logger();

    #[cfg(feature = "dh")]
    let _dh = Dhat::start_heap_profiling();

    let args = Args::parse();

    let shutdown = Arc::new(AtomicBool::new(false));

    dbg!(args);

    let db = match args.kind {
        0 => Arc::new(Db::SledDb(new_sled_db("default_sled".to_string(), &args))),
        1 => Arc::new(Db::RocksDb(new_rocks_db(
            "default_rocksdb".to_string(),
            &args,
        ))),
        _ => panic!("error db kind: {}", args.kind),
    };
    let mut threads = vec![];

    let now = std::time::Instant::now();

    let n_threads = args.threads;

    for i in 0..=n_threads {
        //let tree = tree.clone();
        let db = db.clone();
        let shutdown = shutdown.clone();

        let t = if i == 0 {
            thread::Builder::new()
                .name("reporter".into())
                .spawn(move || report(shutdown))
                .unwrap()
        } else {
            thread::spawn(move || run(args, db, shutdown))
        };

        threads.push(t);
    }

    if let Some(ops) = args.total_ops {
        assert!(!args.burn_in, "don't set both --burn-in and --total-ops");
        while TOTAL.load(Ordering::Relaxed) < ops {
            thread::sleep(std::time::Duration::from_millis(50));
        }
        shutdown.store(true, Ordering::SeqCst);
    } else if !args.burn_in {
        thread::sleep(std::time::Duration::from_secs(args.duration));
        shutdown.store(true, Ordering::SeqCst);
    }

    for t in threads.into_iter() {
        t.join().unwrap();
    }
    let ops = TOTAL.load(Ordering::SeqCst);
    let time = now.elapsed().as_secs() as usize;

    println!(
        "did {} total ops in {} seconds. {} ops/s",
        ops.to_formatted_string(&Locale::en),
        time,
        ((ops * 1_000) / (time * 1_000)).to_formatted_string(&Locale::en)
    );

    #[cfg(feature = "measure_allocs")]
    println!(
        "allocated {} bytes in {} allocations",
        alloc::ALLOCATED_BYTES
            .load(Ordering::Acquire)
            .to_formatted_string(&Locale::en),
        alloc::ALLOCATIONS
            .load(Ordering::Acquire)
            .to_formatted_string(&Locale::en),
    );

    #[cfg(feature = "metrics")]
    sled::print_profile();
}

#[cfg(feature = "logging")]
pub fn setup_logger() {
    use std::io::Write;

    color_backtrace::install();

    fn tn() -> String {
        std::thread::current()
            .name()
            .unwrap_or("unknown")
            .to_owned()
    }

    let mut builder = env_logger::Builder::new();
    builder
        .format(|buf, record| {
            writeln!(
                buf,
                "{:05} {:25} {:10} {}",
                record.level(),
                tn(),
                record.module_path().unwrap().split("::").last().unwrap(),
                record.args()
            )
        })
        .filter(None, log::LevelFilter::Info);

    if let Ok(env) = std::env::var("RUST_LOG") {
        builder.parse_filters(&env);
    }

    let _r = builder.try_init();
}
