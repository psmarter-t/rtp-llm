#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rtp_llm {

class OutputVocabMapping {
public:
    OutputVocabMapping(int64_t full_vocab_size, std::vector<int32_t> local_to_full, std::string config_digest):
        full_vocab_size_(full_vocab_size),
        local_to_full_(std::move(local_to_full)),
        config_digest_(std::move(config_digest)) {
        if (full_vocab_size_ <= 0) {
            throw std::invalid_argument("output vocabulary full size must be positive");
        }
        if (local_to_full_.empty() || local_to_full_.size() >= static_cast<size_t>(full_vocab_size_)) {
            throw std::invalid_argument("output vocabulary mapping must keep a non-empty proper subset");
        }
        if (!std::is_sorted(local_to_full_.begin(), local_to_full_.end())
            || std::adjacent_find(local_to_full_.begin(), local_to_full_.end()) != local_to_full_.end()) {
            throw std::invalid_argument("output vocabulary mapping must be sorted and unique");
        }
        if (local_to_full_.front() < 0 || local_to_full_.back() >= full_vocab_size_) {
            throw std::invalid_argument("output vocabulary mapping contains an out-of-range token id");
        }

        // Penalty handling translates the complete token history every decode step.
        // A dense reverse map keeps that hot path O(1) per token at a 4-byte/V cost.
        full_to_local_.assign(static_cast<size_t>(full_vocab_size_), -1);
        for (size_t local_id = 0; local_id < local_to_full_.size(); ++local_id) {
            full_to_local_[static_cast<size_t>(local_to_full_[local_id])] = static_cast<int32_t>(local_id);
        }
    }

    int64_t fullVocabSize() const {
        return full_vocab_size_;
    }

    size_t size() const {
        return local_to_full_.size();
    }

    const std::vector<int32_t>& localToFull() const {
        return local_to_full_;
    }

    const std::string& configDigest() const {
        return config_digest_;
    }

    bool contains(int32_t full_id) const {
        return toLocal(full_id).has_value();
    }

    std::optional<int32_t> toLocal(int32_t full_id) const {
        if (full_id < 0 || static_cast<int64_t>(full_id) >= full_vocab_size_) {
            return std::nullopt;
        }
        const auto local_id = full_to_local_[static_cast<size_t>(full_id)];
        return local_id >= 0 ? std::optional<int32_t>(local_id) : std::nullopt;
    }

    int32_t toFull(int32_t local_id) const {
        if (local_id < 0 || static_cast<size_t>(local_id) >= local_to_full_.size()) {
            throw std::out_of_range("local output token id is outside the pruned vocabulary");
        }
        return local_to_full_[local_id];
    }

private:
    int64_t              full_vocab_size_;
    std::vector<int32_t> local_to_full_;
    std::vector<int32_t> full_to_local_;
    std::string          config_digest_;
};

using OutputVocabMappingPtr = std::shared_ptr<const OutputVocabMapping>;

}  // namespace rtp_llm
