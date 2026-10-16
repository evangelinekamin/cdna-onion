// Minimal JSON reader for the protocol's control messages (start/stop). Handles
// the full JSON grammar except \u escapes beyond ASCII, which the protocol never
// sends. Host-only.
#pragma once
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

struct Json {
    enum Kind { Null, Bool, Number, String, Array, Object } kind = Null;
    bool boolean = false;
    double number = 0;
    std::string text;
    std::vector<Json> items;
    std::vector<std::pair<std::string, Json>> fields;

    const Json *get(const std::string &key) const {
        for (const auto &f : fields)
            if (f.first == key) return &f.second;
        return nullptr;
    }
};

class JsonReader {
  public:
    explicit JsonReader(const std::string &s) : s_(s) {}

    bool parse(Json &out, std::string &error) {
        if (!value(out, 0) || (skip(), i_ != s_.size())) {
            error = "malformed JSON near offset " + std::to_string(i_);
            return false;
        }
        return true;
    }

  private:
    const std::string &s_;
    size_t i_ = 0;

    void skip() {
        while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' ||
                                  s_[i_] == '\r'))
            i_++;
    }
    bool literal(const char *word) {
        size_t n = std::char_traits<char>::length(word);
        if (s_.compare(i_, n, word) != 0) return false;
        i_ += n;
        return true;
    }
    bool string(std::string &out) {
        if (i_ >= s_.size() || s_[i_] != '"') return false;
        for (i_++; i_ < s_.size(); i_++) {
            char c = s_[i_];
            if (c == '"') { i_++; return true; }
            if (c == '\\') {
                if (++i_ >= s_.size()) return false;
                char e = s_[i_];
                if (e == 'n') out += '\n';
                else if (e == 't') out += '\t';
                else if (e == 'r') out += '\r';
                else if (e == 'b') out += '\b';
                else if (e == 'f') out += '\f';
                else if (e == '"' || e == '\\' || e == '/') out += e;
                else return false;  // \u is never used by the protocol
            } else {
                out += c;
            }
        }
        return false;
    }
    bool value(Json &out, int depth) {
        if (depth > 32) return false;
        skip();
        if (i_ >= s_.size()) return false;
        char c = s_[i_];
        if (c == '{') {
            out.kind = Json::Object;
            i_++;
            skip();
            if (i_ < s_.size() && s_[i_] == '}') { i_++; return true; }
            while (true) {
                std::string key;
                Json v;
                skip();
                if (!string(key)) return false;
                skip();
                if (i_ >= s_.size() || s_[i_++] != ':') return false;
                if (!value(v, depth + 1)) return false;
                out.fields.emplace_back(std::move(key), std::move(v));
                skip();
                if (i_ < s_.size() && s_[i_] == ',') { i_++; continue; }
                if (i_ < s_.size() && s_[i_] == '}') { i_++; return true; }
                return false;
            }
        }
        if (c == '[') {
            out.kind = Json::Array;
            i_++;
            skip();
            if (i_ < s_.size() && s_[i_] == ']') { i_++; return true; }
            while (true) {
                Json v;
                if (!value(v, depth + 1)) return false;
                out.items.push_back(std::move(v));
                skip();
                if (i_ < s_.size() && s_[i_] == ',') { i_++; continue; }
                if (i_ < s_.size() && s_[i_] == ']') { i_++; return true; }
                return false;
            }
        }
        if (c == '"') { out.kind = Json::String; return string(out.text); }
        if (literal("null")) { out.kind = Json::Null; return true; }
        if (literal("true")) { out.kind = Json::Bool; out.boolean = true; return true; }
        if (literal("false")) { out.kind = Json::Bool; return true; }
        const char *start = s_.c_str() + i_;
        char *end = nullptr;
        out.number = std::strtod(start, &end);
        if (end == start) return false;
        out.kind = Json::Number;
        i_ += (size_t)(end - start);
        return true;
    }
};
