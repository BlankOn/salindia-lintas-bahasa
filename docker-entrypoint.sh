#!/bin/sh
# /data comes from the host, so its owner is whoever created it there -- root,
# if Docker made the directory itself for the bind mount. The app runs as uid
# 1000 and cannot write into a root-owned directory, and SQLite's "unable to
# open database file" is the same message for a missing path, so fix the
# ownership here rather than leaving it to a chown someone has to remember.
#
# Started as root: chown the database directory, then drop to uid 1000 for the
# app itself. Started as anyone else (docker run --user, or a compose `user:`),
# there is nothing to fix and nothing to drop -- run as we are.
set -e

DB_DIR=$(dirname "${DB_PATH:-/data/salindia.sqlite3}")

if [ "$(id -u)" = "0" ]; then
    mkdir -p "$DB_DIR"
    # The directory and any database already in it; not -R over the whole
    # mount, which may hold things that are not ours.
    chown salindia:salindia "$DB_DIR"
    for f in "$DB_DIR"/*.sqlite3 "$DB_DIR"/*.sqlite3-wal "$DB_DIR"/*.sqlite3-shm; do
        [ -e "$f" ] && chown salindia:salindia "$f"
    done
    exec setpriv --reuid=1000 --regid=1000 --init-groups "$@"
fi

exec "$@"
