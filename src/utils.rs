pub(crate) fn init_log(level: String) {
    fn detailed_format(
        w: &mut dyn std::io::Write,
        now: &mut flexi_logger::DeferredNow,
        record: &log::Record,
    ) -> crate::Result<(), std::io::Error> {
        let ts = now.format("%Y-%m-%d %H:%M:%S%.3f");
        let level = match record.level() {
            log::Level::Error => "ERROR",
            log::Level::Warn => "WARNING",
            log::Level::Info => "INFO",
            log::Level::Debug => "DEBUG",
            log::Level::Trace => "TRACE",
        };
        let module = record
            .module_path()
            .unwrap_or("<unnamed>")
            .replace("::", ".");
        write!(
            w,
            "{} | {:<8} | {}:{} - {}",
            ts,
            level,
            module,
            record.line().unwrap_or(0),
            record.args()
        )
    }
    flexi_logger::Logger::try_with_str(level)
        .unwrap()
        .format(detailed_format)
        .log_to_stdout()
        .start()
        .unwrap();
}
