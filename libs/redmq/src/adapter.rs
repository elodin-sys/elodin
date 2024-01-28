pub struct StringAdapter<V>(pub V);

impl<E: ToString, V: std::str::FromStr<Err = E>> redis::FromRedisValue for StringAdapter<V> {
    fn from_redis_value(v: &redis::Value) -> redis::RedisResult<Self> {
        let value = redis::from_redis_value::<String>(v)?
            .parse()
            .map_err(|err: E| {
                redis::RedisError::from((
                    redis::ErrorKind::TypeError,
                    "invalid type",
                    err.to_string(),
                ))
            })?;
        Ok(Self(value))
    }
}

impl<V: ToString> redis::ToRedisArgs for StringAdapter<V> {
    fn write_redis_args<W: ?Sized + redis::RedisWrite>(&self, out: &mut W) {
        self.0.to_string().write_redis_args(out)
    }
}

impl<V> std::ops::Deref for StringAdapter<V> {
    type Target = V;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
